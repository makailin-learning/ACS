import torch

from model.model_utils import *
from torchsummary import summary

class ConvBN(nn.Module):
    def __init__(self, in_channels, kernel_size=3, stride=1, padding=1, deploy=False, activation=None):
        super().__init__()
        if activation is None:
            self.activation = nn.Identity()
        else:
            self.activation = activation
        if deploy:
            self.reparam = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=True)
        else:
            self.conv = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
            self.bn = nn.BatchNorm2d(num_features=in_channels)

    def forward(self, x):
        if hasattr(self, 'reparam'):
            return self.activation(self.reparam(x))
        else:
            return self.activation(self.bn(self.conv(x)))

    def switch_to_deploy(self):
        kernel, bias = I_fusebn(self.conv.weight, self.bn)
        self.reparam = nn.Conv2d(in_channels=self.conv.in_channels, out_channels=self.conv.out_channels, kernel_size=self.conv.kernel_size,
                                      stride=self.conv.stride, padding=self.conv.padding, dilation=self.conv.dilation, groups=self.conv.groups, bias=True)
        self.reparam.weight.data = kernel
        self.reparam.bias.data = bias
        for para in self.parameters():
            para.detach_()
        self.__delattr__('conv')
        self.__delattr__('bn')


class ACS(nn.Module):
    def __init__(self, in_channels, kernel_size=3, stride=1, padding=1, deploy=False, activation=None):
        super(ACS,self).__init__()

        self.deploy = deploy
        self.mid_channels = in_channels // 2
        self.kernel_size = kernel_size

        if activation is None:
            self.activation = nn.Identity()
        else:
            self.activation = activation

        assert kernel_size//2==padding,"kernel_size 与 padding 不匹配"

        # 推理时过一个融合分支reparam
        if deploy:
            self.acs_reparam = nn.Conv2d(in_channels=in_channels, out_channels=self.mid_channels,
                                         kernel_size=kernel_size, stride=stride, padding=padding, bias=True)
        else:
            # 1x1 分支
            self.acs_1x1 = nn.Conv2d(in_channels=in_channels, out_channels=self.mid_channels,
                                     kernel_size=1, stride=stride, padding=0, bias=False)
            self.acs_1x1_bn=nn.BatchNorm2d(self.mid_channels)

            # 1x1-bn-3x3分支
            self.acs_3x3 = nn.Sequential()
            self.acs_3x3.add_module("conv1",nn.Conv2d(in_channels=in_channels, out_channels=self.mid_channels,
                                                      kernel_size=1, stride=1, padding=0, bias=False))
            self.acs_3x3.add_module("bn1", nn.BatchNorm2d(self.mid_channels))
            self.acs_3x3.add_module("conv3",nn.Conv2d(in_channels=self.mid_channels,out_channels=self.mid_channels,
                                                      kernel_size=kernel_size, stride=stride, padding=padding, bias=False))
            self.acs_3x3.add_module("bn2", nn.BatchNorm2d(self.mid_channels))

            # 3x3分支
            self.acs_main = nn.Conv2d(in_channels=in_channels, out_channels=self.mid_channels,
                                      kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
            self.acs_main_bn = nn.BatchNorm2d(self.mid_channels)

            # 1x1-bn-avg分支
            self.acs_avg = nn.Sequential()
            self.acs_avg.add_module("conv1",nn.Conv2d(in_channels=in_channels, out_channels=self.mid_channels,
                                                      kernel_size=1, stride=1, padding=0, bias=False))
            self.acs_avg.add_module("bn1",nn.BatchNorm2d(self.mid_channels))
            self.acs_avg.add_module("avg",nn.AvgPool2d(kernel_size=kernel_size, stride=stride, padding=padding))
            self.acs_avg.add_module("bn2", nn.BatchNorm2d(self.mid_channels))

        # 后处理
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        """
        torch.nn.Parameter是继承自torch.Tensor的子类，其主要作用是作为nn.Module中的可训练参数使用。
        它与torch.Tensor的区别就是nn.Parameter会自动被认为是module的可训练参数，即加入到parameter()这个迭代器中去。
        而module中非nn.Parameter()的普通tensor是不在parameter中的。
        """
        # 引入可训练参数
        self.c_score = nn.Parameter(torch.ones(2*in_channels, device=device), requires_grad=True)

    def forward(self,x):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # 注意这里这个坑,h/w是会进行下采样的,采用原来的h/w,在进行batch resize时，会抵充通道数
        b, c, h, w = x.shape

        if hasattr(self, 'acs_reparam'):
            out = self.acs_reparam(x)
        else:

            # 输入乘以融合乘数
            acs_main = self.acs_main(x)
            acs_1x1 = self.acs_1x1(x)
            acs_3x3 = self.acs_3x3(x)
            acs_avg = self.acs_avg(x)
            out = torch.cat((acs_main, acs_1x1, acs_3x3, acs_avg), dim=1)

        conv_1x1 = torch.zeros([c,2*c,1,1],device=device)
        ind = torch.topk(self.c_score, k=c, dim=0, largest=True, sorted=False).indices
        for i in range(c):
            conv_1x1[i,ind[i],0,0]=torch.sigmoid(self.c_score)[ind[i]]    # 第i个卷积核的第index[i]的通道赋值为分数值，其余通道保持为0

        out = F.conv2d(out,conv_1x1)
        out = self.activation(out)

        return out

    # TODO 融合有问题，再阅读DBB代码
    # 注意该函数融合的是所有权重的数据torch.tensor 而不是可训练数据张量 torch.paramtemers.tensor
    def get_eq_kernel_bias(self):

        # main分支  直接乘以融合参数(仅仅权重，偏差不需要)，转换ok
        k_main_fuse, b_main_fuse = I_fusebn(self.acs_main.weight, self.acs_main_bn)

        # 1x1分支   先乘以融合参数(仅仅权重，偏差不需要)，再扩展为3x3格式，转换ok
        # 融合函数返回的是普通张量nn.tensor,不能进行 nn.tensor = nn.parameters.tensor 赋值
        k_1x1, b_1x1_fuse = I_fusebn(self.acs_1x1.weight, self.acs_1x1_bn)
        k_1x1_fuse = VI_multiscale(k_1x1, self.kernel_size)

        # 1x1-3x3分支
        k_1x1_3x3_first, b_1x1_3x3_first = I_fusebn(self.acs_3x3.conv1.weight, self.acs_3x3.bn1)
        k_1x1_3x3_second, b_1x1_3x3_second = I_fusebn(self.acs_3x3.conv3.weight, self.acs_3x3.bn2)
        k_1x1_3x3_fuse, b_1x1_3x3_fuse = III_1x1_3x3(k_1x1_3x3_first, b_1x1_3x3_first,
                                                     k_1x1_3x3_second, b_1x1_3x3_second)

        # 1x1-avg分支
        k_avg = V_avg(self.mid_channels, self.kernel_size).to(k_main_fuse.device)
        k_1x1_avg_second, b_1x1_avg_second = I_fusebn(k_avg, self.acs_avg.bn2)
        k_1x1_avg_first, b_1x1_avg_first = I_fusebn(self.acs_avg.conv1.weight, self.acs_avg.bn1)
        k_1x1_avg_fuse, b_1x1_avg_fuse = III_1x1_3x3(k_1x1_avg_first, b_1x1_avg_first,
                                                     k_1x1_avg_second, b_1x1_avg_second)

        return IV_concat((k_main_fuse, k_1x1_fuse, k_1x1_3x3_fuse, k_1x1_avg_fuse),
                        (b_main_fuse, b_1x1_fuse, b_1x1_3x3_fuse, b_1x1_avg_fuse))

    def switch_to_deploy(self):
        if hasattr(self, 'acs_reparam'):
            return
        kernel, bias = self.get_eq_kernel_bias()

        self.acs_reparam = nn.Conv2d(in_channels=self.acs_main.in_channels,
                                     out_channels=self.acs_main.in_channels,
                                     kernel_size=self.acs_main.kernel_size,
                                     stride=self.acs_main.stride,
                                     padding=self.acs_main.padding,
                                     bias=True)

        self.acs_reparam.weight.data = kernel
        self.acs_reparam.bias.data = bias
        # 去掉原计算图中的反向传播节点，使这些节点的requires_grad=False
        for para in self.parameters():
            para.detach_()

        # delattr函数用于删除类中的属性
        self.__delattr__('acs_main')
        self.__delattr__('acs_avg')
        self.__delattr__('acs_1x1')
        self.__delattr__('acs_3x3')


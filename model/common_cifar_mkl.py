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

# input 1x1-3x3 cat 3x3 cat 1x1-avg - 1x1
class ACS(nn.Module):
    def __init__(self, in_channels, kernel_size=3, stride=1, padding=1, deploy=False, activation=None):
        super(ACS,self).__init__()

        self.deploy = deploy
        self.is_cat = False

        mid_channels = in_channels // 2
        self.kernel_size = kernel_size

        if activation is None:
            self.activation = nn.Identity()
        else:
            self.activation = activation

        assert kernel_size//2==padding,"kernel_size 与 padding 不匹配"

        # 推理时过一个融合分支reparam
        if deploy:
            self.acs_reparam = nn.Conv2d(in_channels=mid_channels, out_channels=mid_channels,
                                         kernel_size=kernel_size, stride=stride, padding=padding, bias=True)
        else:
            # 1x1-bn-3x3-bn分支
            self.acs_3x3 = nn.Sequential()
            self.acs_3x3.add_module("conv1",nn.Conv2d(in_channels=mid_channels, out_channels=mid_channels,
                                                      kernel_size=1, stride=1, padding=0, bias=False))
            self.acs_3x3.add_module('drop1',nn.Dropout2d(p=0.2))
            self.acs_3x3.add_module("bn1", nn.BatchNorm2d(mid_channels))
            self.acs_3x3.add_module("conv3",nn.Conv2d(in_channels=mid_channels,out_channels=mid_channels,
                                                      kernel_size=kernel_size, stride=stride, padding=padding, bias=False))
            self.acs_3x3.add_module('drop3', nn.Dropout2d(p=0.2))
            self.acs_3x3.add_module("bn2", nn.BatchNorm2d(mid_channels))

            # 3x3分支
            self.acs_main = nn.Sequential()
            self.acs_main.add_module('conv', nn.Conv2d(in_channels=mid_channels, out_channels=mid_channels,
                                      kernel_size=kernel_size, stride=stride, padding=padding, bias=False))
            self.acs_main.add_module('drop', nn.Dropout2d(p=0.2))
            self.acs_main.add_module('bn', nn.BatchNorm2d(mid_channels))

            # 1x1-bn-avg分支
            self.acs_avg = nn.Sequential()
            self.acs_avg.add_module("conv1",nn.Conv2d(in_channels=mid_channels, out_channels=mid_channels,
                                                      kernel_size=1, stride=1, padding=0, bias=False))
            self.acs_avg.add_module('drop1', nn.Dropout2d(p=0.2))
            self.acs_avg.add_module("bn1",nn.BatchNorm2d(mid_channels))
            self.acs_avg.add_module("avg",nn.AvgPool2d(kernel_size=kernel_size, stride=stride, padding=padding))
            self.acs_avg.add_module("bn2", nn.BatchNorm2d(mid_channels))
        # 并行分支
        self.sse = nn.Sequential()
        self.sse.add_module('avg', nn.AdaptiveAvgPool2d(output_size=1))
        self.sse.add_module('conv', nn.Conv2d(in_channels=mid_channels, out_channels=mid_channels,
                                              kernel_size=1, stride=1, padding=0, bias=True))
        self.sse.add_module('drop', nn.Dropout2d(p=0.2))
        self.sse.add_module('bn', nn.BatchNorm2d(mid_channels))
        self.sse.add_module('sig', nn.Sigmoid())

        self.bn = nn.BatchNorm2d(in_channels)
        self.avg = nn.AdaptiveAvgPool2d(output_size=1)
        self.drop = nn.Dropout2d(p=0.2)
        """
        torch.nn.Parameter是继承自torch.Tensor的子类，其主要作用是作为nn.Module中的可训练参数使用。
        它与torch.Tensor的区别就是nn.Parameter会自动被认为是module的可训练参数，即加入到parameter()这个迭代器中去。
        而module中非nn.Parameter()的普通tensor是不在parameter中的。
        """
    def acs(self,n,c,y):
        factor = self.avg(y)
        factor = torch.sigmoid(factor)
        top_index = torch.topk(factor,k=c,dim=1).indices
        out_list = []
        for i in range(n):
            z = top_index[i]
            z1 = z.squeeze(dim=0).squeeze(dim=1).squeeze(dim=1).tolist()
            out_y = y[i,z1,:,:] # [c1,h,w]
            out_list.append(out_y)
        out = torch.stack(out_list) # 将列表中的张量拼接，产生一个新维度 [n,c1,h,w]
        return out

    def forward(self,x):
        # 通道切分
        n,c,h,w = x.shape
        c1 = c//2
        x1 = x[:, 0:c1, :, :]
        x2 = x[:, c1:, :, :]

        if hasattr(self, 'acs_reparam'):
            out = self.acs_reparam(x2)
        else:
            acs_main = self.acs_main(x2)
            acs_3x3 = self.acs_3x3(x2)
            acs_avg = self.acs_avg(x2)
            if self.is_cat:
                out = torch.cat((acs_main,acs_3x3,acs_avg),dim=1)
            else:
                out = acs_main+acs_3x3+acs_avg

        # factor1 = self.sse(x1)
        # x1 = factor1 * x1
        if self.is_cat:
            out = self.acs(n,c1,out)

        y = torch.cat((x1,out),dim=1)
        y = self.activation(y)

        return y

    # TODO 融合有问题，再阅读DBB代码
    # 注意该函数融合的是所有权重的数据torch.tensor 而不是可训练数据张量 torch.paramtemers.tensor
    def get_eq_kernel_bias(self):

        # main分支  直接乘以融合参数(仅仅权重，偏差不需要)，转换ok
        k_main_fuse, b_main_fuse = I_fusebn(self.acs_main.weight, self.acs_main_bn)

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
        if self.is_cat:
            return IV_concat((k_main_fuse, k_1x1_3x3_fuse, k_1x1_avg_fuse),
                                (b_main_fuse, b_1x1_3x3_fuse, b_1x1_avg_fuse))
        else:
            return II_addbranch((k_main_fuse, k_1x1_3x3_fuse, k_1x1_avg_fuse),
                                (b_main_fuse, b_1x1_3x3_fuse, b_1x1_avg_fuse))

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
        self.__delattr__('acs_3x3')


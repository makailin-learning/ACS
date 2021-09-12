from model.model_utils import *
from torchsummary import summary

class ConvBN(nn.Module):
    def __init__(self, in_channels, kernel_size, deploy=False, activation=None):
        super().__init__()
        if activation is None:
            self.activation = nn.Identity()
        else:
            self.activation = activation
        if deploy:
            self.reparam = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=kernel_size, stride=1, padding=kernel_size//2, bias=True)
        else:
            self.conv = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=kernel_size, stride=1, padding=kernel_size//2, bias=False)
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
    def __init__(self, in_channels, kernel_size=3, deploy=False, activation=None):
        super(ACS,self).__init__()

        self.deploy = deploy
        self.mid_channels = in_channels // 2
        self.kernel_size = kernel_size

        if activation is None:
            self.activation = nn.Identity()
        else:
            self.activation = activation

        # 推理时过一个融合分支reparam
        if deploy:
            self.acs_reparam = nn.Conv2d(in_channels=in_channels, out_channels=in_channels,
                                         kernel_size=3, stride=1, padding=1, bias=True)
        else:
            # 1x1 分支
            self.acs_1x1 = nn.Conv2d(in_channels=in_channels, out_channels=self.mid_channels,
                                     kernel_size=1, stride=1, padding=0, bias=True)
            # 1x1-bn-3x3分支
            self.acs_3x3 = nn.Sequential()
            self.acs_3x3.add_module("conv1",nn.Conv2d(in_channels=in_channels, out_channels=self.mid_channels,
                                                      kernel_size=1, stride=1, padding=0, bias=False))
            self.acs_3x3.add_module("bn", nn.BatchNorm2d(self.mid_channels))
            self.acs_3x3.add_module("conv3",nn.Conv2d(in_channels=self.mid_channels,out_channels=self.mid_channels,
                                                      kernel_size=3, stride=1, padding=1, bias=True))

            # 3x3分支
            self.acs_main = nn.Conv2d(in_channels=in_channels, out_channels=self.mid_channels,
                                      kernel_size=3, stride=1, padding=1, bias=True)

            # 1x1-bn-avg分支
            self.acs_avg = nn.Sequential()
            self.acs_avg.add_module("conv1",nn.Conv2d(in_channels=in_channels, out_channels=self.mid_channels,
                                                      kernel_size=1, stride=1, padding=0, bias=False))
            self.acs_avg.add_module("bn",nn.BatchNorm2d(self.mid_channels))
            self.acs_avg.add_module("avg",nn.AvgPool2d(kernel_size=3, stride=1, padding=1))

            # 后处理
            self.acs_channel = nn.AdaptiveMaxPool2d(output_size=1)
            self.bn=nn.BatchNorm2d(in_channels)
            self.cat_score = nn.ReLU()

    def forward(self,x):

        b, c, h, w = x.shape

        if not hasattr(self, 'acs_reparam'):

            # 获得拼接乘数
            cat_score = nn.Parameter(torch.ones(4,device=x.device),requires_grad=True)
            self.param = torch.softmax(self.cat_score(cat_score),dim=0)

            # 输入乘以拼接乘数
            acs_main = self.acs_main(x*self.param[0])
            acs_1x1 = self.acs_1x1(x*self.param[1])
            acs_3x3 = self.acs_3x3(x*self.param[2])
            acs_avg = self.acs_avg(x*self.param[3])
            """
            torch.nn.Parameter是继承自torch.Tensor的子类，其主要作用是作为nn.Module中的可训练参数使用。
            它与torch.Tensor的区别就是nn.Parameter会自动被认为是module的可训练参数，即加入到parameter()这个迭代器中去；
            而module中非nn.Parameter()的普通tensor是不在parameter中的。
            """
            out = torch.cat((acs_main,acs_1x1,acs_3x3,acs_avg),dim=1)

            # 通道选择器
            mid_c=out.shape[1]
            c_score = self.acs_channel(out)
            c_score = torch.sigmoid(c_score)
            c_score = c_score.reshape(b,-1)
            c_index = torch.topk(c_score, k=c, dim=1, largest=True, sorted=True).indices
            c_index = c_index.sort(dim=1).values
            c_offset = torch.arange(0, b).reshape(-1, 1)
            c_index = c_index + c_offset * mid_c
            self.c_mask = c_index.reshape(-1)

        else:
            out = self.acs_reparam(x)

        out = out.reshape(-1, h, w)
        out = out[self.c_mask, :, :]
        out = out.reshape(b, -1, h, w)
        out = self.activation(self.bn(out))

        out = out + x

        return out

    # TODO 融合有问题，再阅读DBB代码
    # 注意该函数融合的是所有权重的数据torch.tensor 而不是可训练数据张量 torch.paramtemers.tensor
    def get_eq_kernel_bias(self):

        # main分支  直接乘以融合参数(仅仅权重，偏差不需要)，转换ok
        k_main = self.acs_main.weight.data*self.param[0]
        b_main = self.acs_main.bias.data

        # 1x1分支   先乘以融合参数(仅仅权重，偏差不需要)，再扩展为3x3格式，转换ok
        # 融合函数返回的是普通张量nn.tensor
        k_1x1 = VI_multiscale(self.acs_1x1.weight, self.kernel_size)*self.param[1]
        b_1x1 = self.acs_1x1.bias.data    # 所以不能进行 nn.tensor = nn.parameters.tensor 赋值

        # 1x1-3x3分支
        k_1x1_3x3_first, b_1x1_3x3_first = I_fusebn(self.acs_3x3.conv1.weight, self.acs_3x3.bn)
        k_1x1_3x3_second, b_1x1_3x3_second = self.acs_3x3.conv3.weight.data , self.acs_3x3.conv3.bias.data

        k_1x1_3x3_merged, b_1x1_3x3_merged = III_1x1_3x3(k_1x1_3x3_first*self.param[2], b_1x1_3x3_first,
                                                         k_1x1_3x3_second,b_1x1_3x3_second)

        # 1x1-avg分支
        k_1x1_avg_first, b_1x1_avg_first = I_fusebn(self.acs_avg.conv1.weight, self.acs_avg.bn)
        k_avg = V_avg(self.mid_channels, self.kernel_size)
        k_1x1_avg_second, b_1x1_avg_second = k_avg,torch.zeros(k_avg.shape[0],device=k_avg.device)

        k_1x1_avg_merged, b_1x1_avg_merged = III_1x1_3x3(k_1x1_avg_first*self.param[3], b_1x1_avg_first,
                                                         k_1x1_avg_second, b_1x1_avg_second)


        return IV_concat((k_main, k_1x1, k_1x1_3x3_merged, k_1x1_avg_merged),
                         (b_main, b_1x1, b_1x1_3x3_merged, b_1x1_avg_merged))

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


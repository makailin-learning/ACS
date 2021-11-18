import torch

from model.model_utils import *
from torchsummary import summary

class ConvBN(nn.Module):
    def __init__(self, in_channels, out_channels,kernel_size=3, stride=1, padding=1, deploy=False, activation=None):
        super().__init__()
        if activation is None:
            self.activation = nn.Identity()
        else:
            self.activation = activation
        if deploy:
            self.reparam = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=True)
        else:
            self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
            self.bn = nn.BatchNorm2d(num_features=out_channels)
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

# 在Bottleneck中，不需要调整通道数，仅下采样，在BasicBlock中既要下采样又要提升通道
class ACS(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1, deploy=False, activation=None):
        super(ACS,self).__init__()

        self.deploy = deploy
        mid_ch1 = in_ch // 2
        mid_ch2 = out_ch // 2
        self.kernel_size = kernel_size

        if activation is None:
            self.activation = nn.Identity()
        else:
            self.activation = activation

        assert kernel_size//2 == padding,"kernel_size 与 padding 不匹配"

        # 推理时过一个融合分支reparam
        if deploy:
            self.acs_reparam = nn.Conv2d(in_channels=mid_ch2, out_channels=mid_ch2,
                                         kernel_size=kernel_size, stride=1, padding=padding, bias=True)
        else:
            # # 1x1 分支
            # self.acs_1x1 = nn.Conv2d(in_channels=mid_channels, out_channels=mid_channels,
            #                          kernel_size=1, stride=stride, padding=0, bias=False)
            # self.acs_1x1_bn = nn.BatchNorm2d(mid_channels)

            # 1x1-bn-3x3分支
            self.acs_3x3 = nn.Sequential()
            self.acs_3x3.add_module("conv1",nn.Conv2d(in_channels=mid_ch2, out_channels=mid_ch2,
                                                      kernel_size=1, stride=1, padding=0, bias=False))
            self.acs_3x3.add_module("bn1", nn.BatchNorm2d(mid_ch2))
            self.acs_3x3.add_module("conv3",nn.Conv2d(in_channels=mid_ch2,out_channels=mid_ch2,
                                                      kernel_size=kernel_size, stride=1, padding=padding, bias=False))
            self.acs_3x3.add_module("bn2", nn.BatchNorm2d(mid_ch2))

            # 3x3分支
            self.acs_main = nn.Conv2d(in_channels=mid_ch2, out_channels=mid_ch2,
                                      kernel_size=kernel_size, stride=1, padding=padding, bias=False)
            self.acs_main_bn = nn.BatchNorm2d(mid_ch2)

        # 两个条件同时满足时，才是Basic，不然就是Bottle
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential()
            self.shortcut.add_module('conv', nn.Conv2d(mid_ch1, mid_ch2, kernel_size=1, stride=stride))
            self.shortcut.add_module('bn', nn.BatchNorm2d(mid_ch2))
        else:
            self.shortcut = nn.Identity()

        # 后处理
        self.gap = nn.AdaptiveAvgPool2d(output_size=1)

        self.cse = nn.Sequential()
        self.cse.add_module('conv_down', nn.Conv2d(in_channels=mid_ch2,out_channels=mid_ch2 // 4,
                                               kernel_size=1,stride=1,padding=0,bias=True))
        self.cse.add_module('relu',nn.ReLU())
        self.cse.add_module('conv_up', nn.Conv2d(in_channels=mid_ch2 // 4, out_channels=mid_ch2,
                                               kernel_size=1, stride=1, padding=0, bias=True))
        self.cse.add_module('sig', nn.Sigmoid())

    def shuffle_channel(self,y,group):
        b,c,h,w = y.shape
        y = y.view(b,group,-1,h,w)
        y = y.permute(0,2,1,3,4).contiguous()
        y = y.view(b,-1,h,w)
        return y

    def forward(self,x):
        x = self.shortcut(x)
        b, c, h, w = x.shape
        c1 = c // 2

        x1 = x[:, 0:c1, :, :]
        x2 = x[:, c1:, :, :]
        print(x1.shape, x2.shape)
        factor1 = self.gap(x2)

        if hasattr(self, 'acs_reparam'):
            x2 = self.acs_reparam(x2)
        else:
            acs_main = self.acs_main(x2)
            # acs_1x1 = self.acs_1x1(x2)
            acs_3x3 = self.acs_3x3(x2)
            x2 = acs_main + acs_3x3

        factor2 = self.gap(x2)
        factor = factor2 - factor1
        factor = self.cse(factor)
        x2 = x2 * factor

        y = torch.cat((x1,x2),dim=1)
        y = self.shuffle_channel(y,group=4)
        y = self.activation(y)

        return y

    def get_eq_kernel_bias(self):

        # main分支  直接乘以融合参数(仅仅权重，偏差不需要)，转换ok
        k_main_fuse, b_main_fuse = I_fusebn(self.acs_main.weight, self.acs_main_bn)

        # # 融合函数返回的是普通张量nn.tensor,不能进行 nn.tensor = nn.parameters.tensor 赋值
        # k_1x1, b_1x1_fuse = I_fusebn(self.acs_1x1.weight, self.acs_1x1_bn)
        # k_1x1_fuse = VI_multiscale(k_1x1, self.kernel_size)

        # 1x1-3x3分支
        k_1x1_3x3_first, b_1x1_3x3_first = I_fusebn(self.acs_3x3.conv1.weight, self.acs_3x3.bn1)
        k_1x1_3x3_second, b_1x1_3x3_second = I_fusebn(self.acs_3x3.conv3.weight, self.acs_3x3.bn2)
        k_1x1_3x3_fuse, b_1x1_3x3_fuse = III_1x1_3x3(k_1x1_3x3_first, b_1x1_3x3_first,
                                                     k_1x1_3x3_second, b_1x1_3x3_second)

        return II_addbranch((k_main_fuse, k_1x1_3x3_fuse),
                            (b_main_fuse, b_1x1_3x3_fuse))

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
        # self.__delattr__('acs_1x1')
        # self.__delattr__('acs_avg')
        self.__delattr__('acs_3x3')


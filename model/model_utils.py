import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

#转换一：3x3conv和bn融合
def I_fusebn(kernel, bn):
    #传入的kernel为卷积权重，bn为BN网络层结构
    gamma = bn.weight
    std = (bn.running_var + bn.eps).sqrt()
    #根据repvgg的融合公式
    return kernel * ((gamma / std).reshape(-1, 1, 1, 1)), bn.bias - bn.running_mean * gamma / std

def _fuse_bn_tensor(conv,bn):

    kernel = conv.weight
    running_mean = bn.running_mean
    running_var = bn.running_var
    gamma = bn.weight
    beta = bn.bias
    eps = bn.eps

    std = (running_var + eps).sqrt()
    t = (gamma / std).reshape(-1, 1, 1, 1)
    return kernel * t, beta - running_mean * gamma / std

#转换二：多并行分支，相加融合
def II_addbranch(kernels, biases):
    return sum(kernels), sum(biases)

#转换三：1x1conv和3x3conv
def III_1x1_3x3(k1, b1, k2, b2, groups=1):
    if groups == 1:
        # 例 input：Bx3x3x3-> (k1：2x3x1x1 -> Bx2x3x3 -> k2:2x2x3x3)-> output:Bx2x1x1
        k = F.conv2d(k2, k1.permute(1, 0, 2, 3))   #把k2作为输入，k1调整通道后作为卷积核，进行卷积得到融合卷积核k
        b_hat = (k2 * b1.reshape(1, -1, 1, 1)).sum((1, 2, 3))
    else:
        k_slices = []
        b_slices = []
        k1_T = k1.permute(1, 0, 2, 3)
        k1_group_width = k1.size(0) // groups
        k2_group_width = k2.size(0) // groups
        for g in range(groups):
            k1_T_slice = k1_T[:, g*k1_group_width:(g+1)*k1_group_width, :, :]
            k2_slice = k2[g*k2_group_width:(g+1)*k2_group_width, :, :, :]
            k_slices.append(F.conv2d(k2_slice, k1_T_slice))
            b_slices.append((k2_slice * b1[g*k1_group_width:(g+1)*k1_group_width].reshape(1, -1, 1, 1)).sum((1, 2, 3)))
        k, b_hat = transIV_depthconcat(k_slices, b_slices)
    return k, b_hat + b2

#转换四：concat
def IV_concat(kernels, biases):
    return torch.cat(kernels, dim=0), torch.cat(biases)

#转换五：1x1conv和avg
def V_avg(channels, kernel_size, groups=1):
    input_dim = channels // groups
    k = torch.zeros((channels, input_dim, kernel_size, kernel_size))
    #均值池化没有卷积核，即逐通道上进行均值操作
    #需建立同输出通道数一样多个卷积核，而且通道数也一样,np.tile类同与torch.repeat，数值>=2才进行复制
    k[np.arange(channels), np.tile(np.arange(input_dim), groups), :, :] = 1.0 / kernel_size ** 2
    return k

#转换六：1x1conv和bn融合，最后转化为3x3的kernel_size
def VI_multiscale(kernel, target_kernel_size):
    #kernel=BxCx1x1 target_kernel_size=3
    H_pixels_to_pad = (target_kernel_size - kernel.size(2)) // 2
    W_pixels_to_pad = (target_kernel_size - kernel.size(3)) // 2
    #上下左右各填充1个为0的像素
    #      0 0 0
    # x -> 0 x 0
    #      0 0 0
    return F.pad(kernel, [H_pixels_to_pad, H_pixels_to_pad, W_pixels_to_pad, W_pixels_to_pad])


class IBC1x1(nn.Conv2d):

    def __init__(self, channels, groups=1):
        super(IBC1x1, self).__init__(in_channels=channels, out_channels=channels, kernel_size=1,
                                     stride=1, padding=0, groups=groups, bias=False)
        # 保证通道能够被正确分组，除不尽就报错
        assert channels % groups == 0
        input_dim = channels // groups
        # 创建一个1x1的卷积核[out_ch,in_ch,1,1]
        id_value = np.zeros((channels, input_dim, 1, 1))

        # 第i个张量体的第i个通道为1,其余通道为0，实现identity的卷积过程
        for i in range(channels):
            id_value[i, i % input_dim, 0, 0] = 1

            # tensor.type_as(tensor),使张量获得同另一个张量的相同数据类型
        self.id_tensor = torch.from_numpy(id_value).type_as(self.weight)
        nn.init.zeros_(self.weight)

    def forward(self, input):
        # 融合后的卷积核
        kernel = self.weight + self.id_tensor.to(self.weight.device)
        # F.conv2d就纯粹是个函数，需要手动定义权重和偏差
        result = F.conv2d(input, kernel, None, stride=1, padding=0, dilation=self.dilation, groups=self.groups)
        return result  # result=input

    def get_actual_kernel(self):
        return self.weight + self.id_tensor.to(self.weight.device)


class BPL(nn.Module):
    def __init__(self, pad_pixels, num_features):
        super(BPL, self).__init__()
        self.bn = nn.BatchNorm2d(num_features)
        # 填充像素值=padding
        self.pad_pixels = pad_pixels

    def forward(self, input):
        output = self.bn(input)
        if self.pad_pixels > 0:
            # BN层的数学计算公式,bias为偏差、weight为权重、running_mean为均值、running_val为标准差
            if self.bn.affine:
                # b-mean*W/根号下Var
                pad_values = self.bn.bias.detach() - self.bn.running_mean * self.bn.weight.detach() / \
                             torch.sqrt(self.bn.running_var + self.bn.eps)
            else:
                pad_values = - self.bn.running_mean / torch.sqrt(self.bn.running_var + self.bn.eps)

            # 在output的四周进行数值填充0，pad_pixels表示填充行数和列数1，4表示上下左右四个方向
            output = F.pad(output, [self.pad_pixels] * 4)  # 1x2x3x3->1x2x5x5
            pad_values = pad_values.view(1, -1, 1, 1)  # [x,y,z]->1x3x1x1
            output[:, :, 0:self.pad_pixels, :] = pad_values  # 对填充的上边那些行广播赋值 1x3x1x5=(1x3x1x1->1x3x1x5)
            output[:, :, -self.pad_pixels:, :] = pad_values  # 对填充的下边那些行广播赋值
            output[:, :, :, 0:self.pad_pixels] = pad_values  # 对填充的左边那些行广播赋值 1x3x5x1=(1x3x1x1->1x3x5x1)
            output[:, :, :, -self.pad_pixels:] = pad_values  # 对填充的右边那些行广播赋值

        return output

    # @property是python中的描述符，最大的好处就是在类中把一个方法变成属性调用
    # BPL.weight()是一个成员函数，返回self.bn.weight，而BPL.weight仅仅是成员函数的名称，不会有返回值
    # 使用了property后，就相当于 BPL.weight = BPL.weight() = self.bn.weight
    @property
    def weight(self):
        return self.bn.weight

    @property
    def bias(self):
        return self.bn.bias

    @property
    def running_mean(self):
        return self.bn.running_mean

    @property
    def running_var(self):
        return self.bn.running_var

    @property
    def eps(self):
        return self.bn.eps
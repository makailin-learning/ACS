import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

def weights_init(m):
    if isinstance(m,nn.Linear):
        nn.init.xavier_normal_(m.weight)
        nn.init.constant_(m.bias,0.0)
    elif isinstance(m,nn.Conv2d):
        nn.init.kaiming_normal_(m.weight,mode='fan_out',nonlinearity='relu')
    elif isinstance(m,nn.BatchNorm2d):
        nn.init.uniform_(m.weight,0.02,1.)
        nn.init.constant_(m.bias,0.0)

# 转换一：3x3conv和bn融合
def I_fusebn(kernel, bn):
    # 传入的kernel为卷积权重，bn为BN网络层结构
    gamma = bn.weight
    std = (bn.running_var + bn.eps).sqrt()
    # 根据repvgg的融合公式
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

# 转换二：多并行分支，相加融合
def II_addbranch(kernels, biases):

    return sum(kernels), sum(biases)

# 转换三：1x1conv和3x3conv
def III_1x1_3x3(k1, b1, k2, b2, groups=1):
    if groups == 1:
        # 例 input：Bx3x3x3-> (k1：2x3x1x1 -> Bx2x3x3 -> k2:2x2x3x3)-> output:Bx2x1x1
        k = F.conv2d(k2, k1.permute(1, 0, 2, 3))   # 把k2作为输入，k1调整通道后作为卷积核，进行卷积得到融合卷积核k
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
        k, b_hat = IV_concat(k_slices, b_slices)
    return k, b_hat + b2

# 转换四：concat
def IV_concat(kernels, biases):
    return torch.cat(kernels, dim=0), torch.cat(biases)

# 转换五：1x1conv和avg
def V_avg(channels, kernel_size, groups=1):
    input_dim = channels // groups
    k = torch.zeros((channels, input_dim, kernel_size, kernel_size))
    # 均值池化没有卷积核，即逐通道上进行均值操作
    # 需建立同输出通道数一样多个卷积核，而且通道数也一样,np.tile类同与torch.repeat，数值>=2才进行复制
    k[np.arange(channels), np.tile(np.arange(input_dim), groups), :, :] = 1.0 / kernel_size ** 2
    return k

# 转换六：1x1conv和bn融合，最后转化为3x3的kernel_size
def VI_multiscale(kernel, target_kernel_size):
    # kernel=BxCx1x1 target_kernel_size=3
    h_pixels_to_pad = (target_kernel_size - kernel.size(2)) // 2
    w_pixels_to_pad = (target_kernel_size - kernel.size(3)) // 2
    # 上下左右各填充1个为0的像素
    #      0 0 0
    # x -> 0 x 0
    #      0 0 0
    return F.pad(kernel, [h_pixels_to_pad, h_pixels_to_pad, w_pixels_to_pad, w_pixels_to_pad])


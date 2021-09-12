import torch.nn as nn
import torch.nn.functional as F
from common import *
#from convnet_utils import conv_bn, conv_bn_relu

# (ACS+ACS / ConvBN+ConvBN | shortcut) + relu
class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_ch, out_ch, stride=1, is_acs=False):
        super(BasicBlock, self).__init__()
        if stride != 1 or in_ch != self.expansion * out_ch:
            self.shortcut = nn.Sequential()
            self.shortcut.add_module('conv', nn.Conv2d(in_ch, self.expansion * out_ch, kernel_size=1, stride=stride))
            self.shortcut.add_module('bn', nn.BatchNorm2d(self.expansion * out_ch))
        else:
            self.shortcut = nn.Identity()
        if is_acs:
            self.conv1 = ACS(self.expansion * out_ch, kernel_size=3, deploy=False, activation=nn.ReLU())
            self.conv2 = ACS(self.expansion * out_ch, kernel_size=3, deploy=False, activation=None)
        else:
            self.conv1 = ConvBN(self.expansion * out_ch, kernel_size=3, deploy=False, activation=nn.ReLU())
            self.conv2 = ConvBN(self.expansion * out_ch, kernel_size=3, deploy=False, activation=None)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = out + self.shortcut(x)
        out = F.relu(out)
        return out

# (1x1 + ACS/ConvBN + 1X1 | shortcut) + relu
class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, in_ch, out_ch, stride=1, is_acs=False):
        super(Bottleneck, self).__init__()

        # stride不为1即img_size下采样,每个block结束时输出升维4倍
        if stride != 1 or in_ch != self.expansion*out_ch:
            self.shortcut =nn.Sequential()
            self.shortcut.add_module('conv',nn.Conv2d(in_ch, self.expansion*out_ch, kernel_size=1, stride=stride))
            self.shortcut.add_module(('bn',nn.BatchNorm2d(self.expansion*out_ch)))
        else:
            self.shortcut = nn.Identity()
        # 1x1 4c-c
        self.conv1 = nn.Sequential()
        self.conv1.add_module('conv1_1',nn.Conv2d(in_ch,out_ch,kernel_size=1))
        self.conv1.add_module('bn',nn.BatchNorm2d(out_ch))
        self.conv1.add_module('relu',nn.ReLU())
        # 3x3 c-c
        if is_acs:
            self.conv2 = ACS(out_ch, kernel_size=3, deploy=False, activation=nn.ReLU())
        else:
            self.conv2 = ConvBN(out_ch,kernel_size=3,deploy=False, activation=nn.ReLU())
        # 1x1 c-4c
        self.conv3 = nn.Sequential()
        self.conv3.add_module('conv1_1', nn.Conv2d(out_ch, self.expansion*out_ch, kernel_size=1))
        self.conv3.add_module('bn', nn.BatchNorm2d(self.expansion*out_ch))

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=1000, width_multiplier=1, is_acs=False):
        super(ResNet, self).__init__()

        self.in_ch = int(64 * width_multiplier)
        self.is_acs = is_acs

        # 模块化之前：7x7 卷积一个，3x3最大池化一个,步距均为2
        self.stage0 = nn.Sequential()
        self.stage0.add_module('conv1', nn.Conv2d(in_channels=3, out_channels=self.in_ch, kernel_size=7, stride=2, padding=3))
        self.stage0.add_module('bn',nn.BatchNorm2d(self.in_ch))
        self.stage0.add_module('relu',nn.ReLU(inplace=True))
        self.stage0.add_module('maxpool', nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

        self.stage1 = self._make_stage(block, int(64 * width_multiplier), num_blocks[0], stride=1)
        self.stage2 = self._make_stage(block, int(128 * width_multiplier), num_blocks[1], stride=2)
        self.stage3 = self._make_stage(block, int(256 * width_multiplier), num_blocks[2], stride=2)
        self.stage4 = self._make_stage(block, int(512 * width_multiplier), num_blocks[3], stride=2)

        # 目标检测时不要这两行
        self.gap = nn.AdaptiveAvgPool2d(output_size=1)
        self.linear = nn.Linear(int(512*block.expansion*width_multiplier), num_classes)

    def _make_stage(self, block, out_ch, num_blocks, stride):
        # 步距list[2,1,1,1] 先降采样
        strides = [stride] + [1]*(num_blocks-1)
        blocks = []

        for stride in strides:
            if block is Bottleneck:  # 1x1 c 3x3 c 1x1 4c
                blocks.append(block(in_ch=self.in_ch, out_ch=int(out_ch), stride=stride, is_acs=self.is_acs))
            else:
                blocks.append(block(in_ch=self.in_ch, out_ch=int(out_ch), stride=stride, is_acs=self.is_acs))
            self.in_ch = int(out_ch * block.expansion)
        return nn.Sequential(*blocks)

    def forward(self, x):
        out = self.stage0(x)
        out = self.stage1(out)
        out = self.stage2(out)
        out = self.stage3(out)
        out = self.stage4(out)
        out = self.gap(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

# 普通resnet
def create_Res18():
    return ResNet(BasicBlock, [2,2,2,2], num_classes=1000, width_multiplier=1, is_acs=False)

def create_Res50():
    return ResNet(Bottleneck, [3,4,6,3], num_classes=1000, width_multiplier=1, is_acs=False)

def create_Res101():
    return ResNet(Bottleneck, [3,4,23,3], num_classes=1000, width_multiplier=1, is_acs=False)

# acs_resnet
def create_Acs_Res18():
    return ResNet(BasicBlock, [2,2,2,2], num_classes=1000, width_multiplier=1, is_acs=True)

def create_Acs_Res50():
    return ResNet(Bottleneck, [3,4,6,3], num_classes=1000, width_multiplier=1, is_acs=True)

def create_Acs_Res101():
    return ResNet(Bottleneck, [3,4,23,3], num_classes=1000, width_multiplier=1, is_acs=True)
from model.common_new import *

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

# ACS/ConvBN + relu
class Bottleneck(nn.Module):
    def __init__(self, in_ch, stride=1, is_acs=False):
        super(Bottleneck, self).__init__()
        self.in_ch=in_ch

        if is_acs:
            self.conv1 = ACS(in_ch, kernel_size=3, stride=stride, padding=1, deploy=False, activation=nn.ReLU())
        else:
            self.conv1 = ConvBN(in_ch, kernel_size=3, stride=stride, padding=1, deploy=False, activation=nn.ReLU())

    def forward(self, x):
        out = self.conv1(x)
        return out

class Down(nn.Module):
    def __init__(self, in_ch, width_multiplier):
        super(Down, self).__init__()
        self.conv = nn.Sequential()
        self.conv.add_module('conv', nn.Conv2d(in_channels=int(in_ch * width_multiplier),
                                                   out_channels=int(in_ch * 2 * width_multiplier),
                                                   kernel_size=1, stride=1, padding=0))
        self.conv.add_module('bn', nn.BatchNorm2d(int(in_ch * 2 * width_multiplier)))
        self.conv.add_module('relu', nn.ReLU(inplace=True))
        self.conv.add_module('maxpool', nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

    def forward(self,x):
        return self.conv(x)

class Line(nn.Module):
    def __init__(self, in_ch, width_multiplier):
        super(Line, self).__init__()
        self.line = nn.Sequential()
        self.line.add_module('conv1', nn.Conv2d(in_channels=int(in_ch * width_multiplier),
                                                 out_channels=int(in_ch * 2 * width_multiplier), kernel_size=1, stride=1,
                                                 padding=0))
        self.line.add_module('bn', nn.BatchNorm2d(int(in_ch * 2 * width_multiplier)))
        self.line.add_module('relu', nn.ReLU(inplace=True))
        self.line.add_module('maxpool', nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

    def forward(self,x):
        return self.line(x)

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=1000, width_multiplier=1., is_acs=False):
        super(ResNet, self).__init__()

        self.in_ch = int(64 * width_multiplier)
        self.is_acs = is_acs
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 模块化之前：7x7 卷积一个，3x3最大池化一个,步距均为2
        self.stage0 = nn.Sequential()
        self.stage0.add_module('conv1', nn.Conv2d(in_channels=3, out_channels=self.in_ch, kernel_size=7, stride=1, padding=3))
        self.stage0.add_module('bn',nn.BatchNorm2d(self.in_ch))
        self.stage0.add_module('relu',nn.ReLU(inplace=True))
        self.stage0.add_module('maxpool', nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        # 7x7卷积扩展通道数3->64 ，最大池化降采样 32->16

        # 卷积扩展通道数64->128 ， 最大池化降采样 16->8
        self.line1 = Line(64, width_multiplier)
        # 卷积扩展通道数128->256 ，最大池化降采样 8->4
        self.line2 = Line(128, width_multiplier)
        # 卷积扩展通道数256->512 ， 最大池化降采样 4->2
        self.line3 = Line(256, width_multiplier)

        self.stage1 = self._make_stage(block, int(128 * width_multiplier), num_blocks[0], stride=1)
        self.stage2 = self._make_stage(block, int(256 * width_multiplier), num_blocks[1], stride=1)
        self.stage3 = self._make_stage(block, int(512 * width_multiplier), num_blocks[2], stride=1)

        # 融合之前的通道改变与降采样
        self.conv_8_4 = Down(128, width_multiplier)
        self.conv_4_2 = Down(256, width_multiplier)
        self.conv_2_1 = Down(512, width_multiplier)

        self.fuse_weight1 = nn.Parameter(torch.ones(2, device=device), requires_grad=True)
        self.fuse_weight2 = nn.Parameter(torch.ones(2, device=device), requires_grad=True)

        # 目标检测时不要这两行
        self.gap = nn.AdaptiveAvgPool2d(output_size=1)
        self.linear = nn.Linear(int(1024*width_multiplier), num_classes)

    def _make_stage(self, block, in_ch, num_blocks, stride):
        # 步距list[1,1,1,1]
        strides = [stride] + [1]*(num_blocks-1)
        blocks = []

        for stride in strides:
            if block is Bottleneck:
                blocks.append(block(in_ch=in_ch, stride=stride, is_acs=self.is_acs))
            else:
                blocks.append(block(in_ch=in_ch, stride=stride, is_acs=self.is_acs))
        return nn.Sequential(*blocks)

    def forward(self, x):

        weight1 = torch.softmax(self.fuse_weight1, dim=0)
        weight2 = torch.softmax(self.fuse_weight2, dim=0)

        # 1x3x32x32->1x64x16x16
        out0 = self.stage0(x)
        # 1x64x16x16->1x128x8x8、1x256x4x4、1x512x2x2
        out1 = self.line1(out0)
        out2 = self.line2(out1)
        out3 = self.line3(out2)
        # 1x128x8x8->1x256x4x4
        out1 = self.stage1(out1)
        out1 = self.conv_8_4(out1)
        # 1x256x4x4 + 1x256x4x4 -> 1x512x2x2
        out2 = self.stage2(out2)
        out2 = out1*weight1[0] + out2*weight1[1]
        out2 = self.conv_4_2(out2)
        # 1x512x2x2 + 1x512x2x2
        out3 = self.stage3(out3)
        out3 = out2*weight2[0] + out3*weight2[0]
        out3 = self.conv_2_1(out3)

        out = self.gap(out3)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        with open('E:/ACS/weight1.txt', 'w') as f:
            print(weight1, file=f)
        with open('E:/ACS/weight2.txt', 'w') as f:
            print(weight2, file=f)
        return out

def Acs_Res18_s(is_acs=False):
    return ResNet(Bottleneck, [4,5,5], num_classes=100, width_multiplier=1, is_acs=is_acs)

def Acs_Res50_l(is_acs=False):
    return ResNet(Bottleneck, [4,5,5], num_classes=100, width_multiplier=1, is_acs=is_acs)

def Acs_Res101(is_acs=False):
    return ResNet(Bottleneck, [4,8,8], num_classes=100, width_multiplier=1, is_acs=is_acs)

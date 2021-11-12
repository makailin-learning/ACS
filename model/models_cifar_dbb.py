import torch.nn

from model.common_cifar_dbb import *

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
            self.conv1 = ACS(self.expansion * out_ch, kernel_size=3, deploy=False, activation=nn.SiLU())
            self.conv2 = ACS(self.expansion * out_ch, kernel_size=3, deploy=False, activation=None)
        else:
            self.conv1 = ConvBN(self.expansion * out_ch, kernel_size=3, deploy=False, activation=nn.SiLU())
            self.conv2 = ConvBN(self.expansion * out_ch, kernel_size=3, deploy=False, activation=None)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = out + self.shortcut(x)
        out = F.silu(out)
        return out

# ACS/ConvBN + relu
class Bottleneck(nn.Module):
    def __init__(self, in_ch, stride=1, is_acs=False):
        super(Bottleneck, self).__init__()
        self.in_ch=in_ch

        if is_acs:
            self.conv1 = ACS(in_ch, kernel_size=3, stride=stride, padding=1, deploy=False, activation=nn.SiLU())
        else:
            self.conv1 = ConvBN(in_ch, kernel_size=3, stride=stride, padding=1, deploy=False, activation=nn.SiLU())

    def forward(self, x):
        out = self.conv1(x)
        return out

class Down(nn.Module):
    def __init__(self, in_ch, width_multiplier):
        super(Down, self).__init__()
        self.downs = nn.Sequential()
        self.downs.add_module('conv', nn.Conv2d(in_channels=int(in_ch * width_multiplier),
                                                   out_channels=int(in_ch * 2 * width_multiplier),
                                                   kernel_size=1, stride=1, padding=0, bias=True))
        self.downs.add_module('bn', nn.BatchNorm2d(int(in_ch * 2 * width_multiplier)))
        self.downs.add_module('silu', nn.SiLU(inplace=True))
        self.downs.add_module('avgpool', nn.AvgPool2d(kernel_size=2, stride=2, padding=0))

        # 分支1 1x1
        self.conv1 = nn.Sequential()
        # self.conv1.add_module('avg',nn.AvgPool2d(kernel_size=2,stride=1))
        self.conv1.add_module('conv', nn.Conv2d(in_channels=int(in_ch * 4 * width_multiplier),
                                                out_channels=int(in_ch * 2 * width_multiplier),
                                                kernel_size=1, stride=1, padding=0, bias=True))
        self.conv1.add_module('bn', nn.BatchNorm2d(int(in_ch * 2 * width_multiplier)))
        # 分支2 3x3
        self.conv3 = nn.Sequential()
        self.conv3.add_module('conv', nn.Conv2d(in_channels=int(in_ch * 4 * width_multiplier),
                                                out_channels=int(in_ch * 2 * width_multiplier),
                                                kernel_size=3, stride=1, padding=1, bias=True))
        self.conv3.add_module('bn', nn.BatchNorm2d(int(in_ch * 2 * width_multiplier)))
        # 分支3 通道加权
        self.sse = nn.Sequential()
        self.sse.add_module('avg', nn.AdaptiveAvgPool2d(output_size=1))
        self.sse.add_module('conv', nn.Conv2d(in_channels=int(in_ch * 4 * width_multiplier),
                                              out_channels=int(in_ch * 2 * width_multiplier),
                                              kernel_size=1, stride=1, padding=0, bias=True))
        self.sse.add_module('sig', nn.Sigmoid())
        # 分支融合后激活
        self.ac = nn.SiLU()

    def forward(self,x,y=None):
        x = self.downs(x)
        if y is not None:
            x = torch.cat((x,y),dim=1)
            x1 = self.conv1(x)
            x2 = self.conv3(x)
            factor = self.sse(x)
            x = x1 + x2
            x = x * factor
            x = self.ac(x)
        return x

class Up(nn.Module):
    def __init__(self,in_ch,width_multiplier):
        super(Up, self).__init__()
        # 调整底层输入的通道数和尺寸
        self.ups = nn.Sequential()
        self.ups.add_module('conv', nn.Conv2d(in_channels=int(in_ch * width_multiplier),
                                                   out_channels=int(in_ch // 2 * width_multiplier),
                                                   kernel_size=1,stride=1,padding=0,bias=True))
        self.ups.add_module('bn', nn.BatchNorm2d(int(in_ch // 2 * width_multiplier)))
        self.ups.add_module('silu', nn.SiLU(inplace=True))
        # 分支1 过卷积
        self.conv3 = nn.Sequential()
        self.conv3.add_module('conv', nn.Conv2d(in_channels=int(in_ch * width_multiplier),
                                                out_channels=int(in_ch // 2 * width_multiplier),
                                                kernel_size=3, stride=1, padding=1, bias=True))
        self.conv3.add_module('bn', nn.BatchNorm2d(int(in_ch // 2 * width_multiplier)))
        # 分支2 通道加权
        self.sse = nn.Sequential()
        self.sse.add_module('avg', nn.AdaptiveAvgPool2d(output_size=1))
        self.sse.add_module('conv', nn.Conv2d(in_channels=int(in_ch * width_multiplier),
                                              out_channels=int(in_ch // 2 * width_multiplier),
                                              kernel_size=1, stride=1, padding=0, bias=True))
        self.sse.add_module('sig', nn.Sigmoid())
        # 分支融合后激活
        self.ac = nn.SiLU()
    def forward(self,x,y=None):
        x = self.ups(x)
        x = F.interpolate(x, scale_factor=2)
        if y is not None:
            x = torch.cat((x,y),dim=1)
            factor = self.sse(x)
            x = self.conv3(x)
            x = x * factor
            x = self.ac(x)
        return x

class Line(nn.Module):
    def __init__(self, in_ch, width_multiplier):
        super(Line, self).__init__()
        self.line = nn.Sequential()
        self.line.add_module('conv1', nn.Conv2d(in_channels=int(in_ch * width_multiplier),
                                                 out_channels=int(in_ch * 2 * width_multiplier), kernel_size=1, stride=1,
                                                 padding=0))
        self.line.add_module('bn', nn.BatchNorm2d(int(in_ch * 2 * width_multiplier)))
        self.line.add_module('silu', nn.SiLU(inplace=True))
        self.line.add_module('maxpool', nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

    def forward(self,x):
        return self.line(x)

class ResNet(nn.Module):
    def __init__(self, block, num_blocks1, num_blocks2, num_classes=1000, width_multiplier=1., is_acs=False):
        super(ResNet, self).__init__()

        self.in_ch = int(64 * width_multiplier)
        self.is_acs = is_acs
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 模块化之前：7x7 卷积一个，3x3最大池化一个,步距均为2
        self.stage0 = nn.Sequential()
        self.stage0.add_module('conv1', nn.Conv2d(in_channels=3, out_channels=self.in_ch, kernel_size=7, stride=1, padding=3))
        self.stage0.add_module('bn',nn.BatchNorm2d(self.in_ch))
        self.stage0.add_module('silu', nn.SiLU(inplace=True))
        self.stage0.add_module('maxpool', nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        # 7x7卷积扩展通道数3->64 ，最大池化降采样 32->16

        # 卷积扩展通道数64->128 ， 最大池化降采样 16->8
        self.line1 = Line(64, width_multiplier)
        # 卷积扩展通道数128->256 ，最大池化降采样 8->4
        self.line2 = Line(128, width_multiplier)
        # 卷积扩展通道数256->512 ， 最大池化降采样 4->2
        self.line3 = Line(256, width_multiplier)

        self.stage1 = self._make_stage(block, int(128 * width_multiplier), num_blocks1[0], stride=1) # 2个1x128x8x8
        self.stage2 = self._make_stage(block, int(256 * width_multiplier), num_blocks1[1], stride=1) # 2个1x256x4x4
        self.stage3 = self._make_stage(block, int(512 * width_multiplier), num_blocks1[2], stride=1) # 1个1x512x2x2

        # FPN
        self.up_2_4 = Up(512,width_multiplier)
        self.up_4_8 = Up(256, width_multiplier)

        self.stage4 = self._make_stage(block, int(128 * width_multiplier), num_blocks2[0], stride=1)  # 2个1x128x8x8
        self.stage5 = self._make_stage(block, int(256 * width_multiplier), num_blocks2[1], stride=1)  # 2个1x256x4x4
        self.stage6 = self._make_stage(block, int(512 * width_multiplier), num_blocks2[2], stride=1)  # 2个1x512x2x2

        # PAN
        self.down_8_4 = Down(128, width_multiplier)
        self.down_4_2 = Down(256, width_multiplier)

        self.fuse_up = nn.Parameter(torch.ones(2, device=device), requires_grad=True)
        self.fuse_down = nn.Parameter(torch.ones(2, device=device), requires_grad=True)

        # 后处理
        self.down_2_1 = Down(512, width_multiplier)
        self.drop = nn.Dropout2d(p=0.2)
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

        # 1x3x32x32->1x64x16x16
        out0 = self.stage0(x)
        # 1x64x16x16->1x128x8x8、1x256x4x4、1x512x2x2
        out1 = self.line1(out0)
        out2 = self.line2(out1)
        out3 = self.line3(out2)
        # 1x128x8x8
        out1 = self.stage1(out1)
        # 1x256x4x4
        out2 = self.stage2(out2)
        # 1x512x2x2
        out3 = self.stage3(out3)

        # FPN
        out3to2 = self.up_2_4(x=out3*self.fuse_up[0],y=out2)
        out1 = self.up_4_8(x=out3to2*self.fuse_up[1],y=out1)

        out1 = self.stage4(out1)
        out2 = self.stage5(out2)
        out3 = self.stage6(out3)

        # PAN
        out1to2 = self.down_8_4(x=out1*self.fuse_down[0],y=out2)
        out3 = self.down_4_2(x=out1to2*self.fuse_down[1],y=out3)

        out3 = self.down_2_1(x=out3)
        out = self.gap(out3)
        out = out.view(out.size(0), -1)

        out = self.linear(out)
        out = self.drop(out)
        # with open('E:/ACS/weight1.txt', 'w') as f:
        #     print(w_up, file=f)
        # with open('E:/ACS/weight2.txt', 'w') as f:
        #     print(w_down, file=f)
        return out

def Acs_Res18_s(is_acs=False):
    return ResNet(Bottleneck, [2,2,1], [2,2,2], num_classes=100, width_multiplier=1, is_acs=is_acs)

def Acs_Res50_l(is_acs=False):
    return ResNet(Bottleneck, [4,5,5], num_classes=100, width_multiplier=1, is_acs=is_acs)

def Acs_Res101(is_acs=False):
    return ResNet(Bottleneck, [4,8,8], num_classes=100, width_multiplier=1, is_acs=is_acs)

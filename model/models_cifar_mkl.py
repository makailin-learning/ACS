import torch.nn

from model.common_cifar_mkl import *

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
    def __init__(self, in_ch, width_multiplier, out_ch=None):
        super(Down, self).__init__()

        self.in_ch = int(in_ch * width_multiplier)
        if out_ch is None:
            self.out_ch = int(in_ch * 2 * width_multiplier)
        else:
            self.out_ch = out_ch
        # ??????1 1x1
        self.conv1 = nn.Sequential()
        self.conv1.add_module('avg',nn.AvgPool2d(kernel_size=2,stride=2))
        self.conv1.add_module('conv', nn.Conv2d(in_channels=self.in_ch,
                                                out_channels=self.out_ch,
                                                kernel_size=1, stride=1, padding=0, bias=True))
<<<<<<< HEAD
        # self.conv1.add_module('drop', nn.Dropout2d(p=0.2))
=======
>>>>>>> 6229e39838423396b703141eb8cbb404f6c79090
        self.conv1.add_module('bn', nn.BatchNorm2d(self.out_ch))

        # ??????2 3x3
        self.conv3 = nn.Sequential()
        self.conv3.add_module('conv', nn.Conv2d(in_channels=self.in_ch,
                                                out_channels=self.out_ch,
                                                kernel_size=3, stride=2, padding=1, bias=True))
<<<<<<< HEAD
        # self.conv3.add_module('drop', nn.Dropout2d(p=0.2))
=======
>>>>>>> 6229e39838423396b703141eb8cbb404f6c79090
        self.conv3.add_module('bn', nn.BatchNorm2d(self.out_ch))

        # ??????3 ????????????
        self.sse = nn.Sequential()
        self.sse.add_module('avg', nn.AdaptiveAvgPool2d(output_size=1))
        self.sse.add_module('conv', nn.Conv2d(in_channels=self.in_ch,
                                              out_channels=self.out_ch,
                                              kernel_size=1, stride=1, padding=0, bias=True))
<<<<<<< HEAD
        self.sse.add_module('drop', nn.Dropout2d(p=0.2))
=======
>>>>>>> 6229e39838423396b703141eb8cbb404f6c79090
        self.sse.add_module('bn',nn.BatchNorm2d(self.out_ch))
        self.sse.add_module('sig', nn.Sigmoid())

        # ?????????????????????
        self.ac = nn.SiLU()

    def forward(self,x):
        x1 = self.conv1(x)
        x2 = self.conv3(x)
        factor = self.sse(x)
        x = x1 + x2
        x = x * factor
        x = self.ac(x)
        return x

class Down_Fuse(nn.Module):
<<<<<<< HEAD
    def __init__(self, in_ch, width_multiplier, is_down = False):
=======
    def __init__(self, in_ch, width_multiplier):
>>>>>>> 6229e39838423396b703141eb8cbb404f6c79090
        super(Down_Fuse, self).__init__()

        self.in_ch = int(in_ch * 2 * width_multiplier)
        # ??????1 1x1
        self.conv1 = nn.Sequential()
<<<<<<< HEAD
        if is_down:
            self.conv1.add_module('avg',nn.AvgPool2d(kernel_size=2,stride=2))
        self.conv1.add_module('conv', nn.Conv2d(in_channels=self.in_ch,
                                                out_channels=self.in_ch,
                                                kernel_size=1, stride=1, padding=0, bias=True))
        self.conv1.add_module('drop', nn.Dropout2d(p=0.2))
        self.conv1.add_module('bn', nn.BatchNorm2d(self.in_ch))
        # ??????2 3x3
        self.conv3 = nn.Sequential()
        if is_down:
            self.conv3.add_module('conv', nn.Conv2d(in_channels=self.in_ch,
                                                    out_channels=self.in_ch,
                                                    kernel_size=3, stride=2, padding=1, bias=True))
        else:
            self.conv3.add_module('conv', nn.Conv2d(in_channels=self.in_ch,
                                                    out_channels=self.in_ch,
                                                    kernel_size=3, stride=1, padding=1, bias=True))
        self.conv3.add_module('drop', nn.Dropout2d(p=0.2))
=======
        self.conv1.add_module('avg',nn.AvgPool2d(kernel_size=2,stride=2))
        self.conv1.add_module('conv', nn.Conv2d(in_channels=self.in_ch,
                                                out_channels=self.in_ch,
                                                kernel_size=1, stride=1, padding=0, groups=2, bias=True))
        self.conv1.add_module('bn', nn.BatchNorm2d(self.in_ch))
        # ??????2 3x3
        self.conv3 = nn.Sequential()
        self.conv3.add_module('conv', nn.Conv2d(in_channels=self.in_ch,
                                                out_channels=self.in_ch,
                                                kernel_size=3, stride=2, padding=1, groups=2, bias=True))
>>>>>>> 6229e39838423396b703141eb8cbb404f6c79090
        self.conv3.add_module('bn', nn.BatchNorm2d(self.in_ch))
        # ??????3 ????????????
        self.sse = nn.Sequential()
        self.sse.add_module('avg', nn.AdaptiveAvgPool2d(output_size=1))
        self.sse.add_module('conv', nn.Conv2d(in_channels=self.in_ch,
                                              out_channels=self.in_ch,
<<<<<<< HEAD
                                              kernel_size=1, stride=1, padding=0, bias=True))
        self.sse.add_module('drop', nn.Dropout2d(p=0.2))
=======
                                              kernel_size=1, stride=1, padding=0, groups=2, bias=True))
>>>>>>> 6229e39838423396b703141eb8cbb404f6c79090
        self.sse.add_module('bn', nn.BatchNorm2d(self.in_ch))
        self.sse.add_module('sig', nn.Sigmoid())
        # ?????????????????????
        self.ac = nn.SiLU()

    def forward(self,x,y):
        if y is None:
<<<<<<< HEAD
            print('??????????????????????????????y')
=======
            print('??????????????????????????????y')
>>>>>>> 6229e39838423396b703141eb8cbb404f6c79090

        x = torch.cat((x,y),dim=1)
        x1 = self.conv1(x)
        x2 = self.conv3(x)
        factor = self.sse(x)
        x = x1 + x2
        x = x * factor
        x = self.ac(x)
        return x

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=1000, width_multiplier=1., is_acs=False):
        super(ResNet, self).__init__()

        self.in_ch = int(64 * width_multiplier)
        self.is_acs = is_acs

<<<<<<< HEAD
        self.stage0 = Down(3,width_multiplier,64)
        # 7x7?????????????????????3->64 ???????????????????????? 32->16

        # ?????????????????????64->128 ??? ????????????????????? 16->8
        self.stream1 = Down(64, width_multiplier)
        # ?????????????????????128->256 ???????????????????????? 8->4
        self.stream2 = Down(128, width_multiplier)
        # ?????????????????????256->512 ??? ????????????????????? 4->2
        self.stream3 = Down(256, width_multiplier)

        self.stage1 = self._make_stage(block, int(128 * width_multiplier), num_blocks[0], stride=1) # 2???1x128x8x8
        self.stage2 = self._make_stage(block, int(256 * width_multiplier), num_blocks[1], stride=1) # 2???1x256x4x4
        self.stage3 = self._make_stage(block, int(512 * width_multiplier), num_blocks[2], stride=1) # 1???1x512x2x2

        self.down_8_4 = Down(128, width_multiplier)
        self.down_4_2 = Down_Fuse(256, width_multiplier, is_down=True)  # ??????????????????????????? 1x256x28x28 - 1x512x14x14
        self.down_2_1 = Down_Fuse(512, width_multiplier, is_down=False) # ????????????????????????   1x512x14x14 - 1x1024x14x14
        self.down_1_0 = Down(1024,width_multiplier)
=======
        self.stage0 = Down(3,width_multiplier,128)
        # 7x7?????????????????????3->64 ???????????????????????? 32->16

        # ?????????????????????64->128 ??? ????????????????????? 16->8
        self.stream1 = Down(128, width_multiplier)
        # ?????????????????????128->256 ???????????????????????? 8->4
        self.stream2 = Down(256, width_multiplier)
        # ?????????????????????256->512 ??? ????????????????????? 4->2
        self.stream3 = Down(512, width_multiplier)

        self.stage1 = self._make_stage(block, int(256 * width_multiplier), num_blocks[0], stride=1) # 2???1x128x8x8
        self.stage2 = self._make_stage(block, int(512 * width_multiplier), num_blocks[1], stride=1) # 2???1x256x4x4
        self.stage3 = self._make_stage(block, int(1024 * width_multiplier), num_blocks[2], stride=1) # 1???1x512x2x2

        self.down_8_4 = Down(256, width_multiplier)
        self.down_4_2 = Down_Fuse(512, width_multiplier)
        self.down_2_1 = Down_Fuse(1024, width_multiplier)
        # self.down_1_0 = Down(2048,width_multiplier)
>>>>>>> 6229e39838423396b703141eb8cbb404f6c79090

        # ?????????
        self.drop = nn.Dropout2d(p=0.2)
        # ??????????????????????????????
        self.gap = nn.AdaptiveAvgPool2d(output_size=1)
        self.linear = nn.Linear(int(2048*width_multiplier), num_classes)

    def _make_stage(self, block, in_ch, num_blocks, stride):
        # ??????list[1,1,1,1]
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
        # 1x64x16x16->1x128x8x8???1x256x4x4???1x512x2x2
        out1 = self.stream1(out0)
        out2 = self.stream2(out1)
        out3 = self.stream3(out2)
        # 1x128x8x8
        out1 = self.stage1(out1)
        # 1x256x4x4
        out2 = self.stage2(out2)
        # 1x512x2x2
        out3 = self.stage3(out3)

        out1 = self.down_8_4(out1)
        fuse_1_2 = self.down_4_2(out2,out1)
        fuse_2_3 = self.down_2_1(out3,fuse_1_2)
<<<<<<< HEAD
        fuse_3_4 = self.down_1_0(fuse_2_3)

        out = self.gap(fuse_3_4)
        out = out.view(out.size(0), -1)
        out = self.drop(out)
        out = self.linear(out)

        return out

def Acs_Res18_s(block=Bottleneck,num_blocks=[4,5,5],num_class=1000,is_acs=False):
    return ResNet(block=block, num_blocks=num_blocks, num_classes=num_class, width_multiplier=1, is_acs=is_acs)
=======
        # out = self.down_1_0(fuse_2_3)

        out = self.gap(fuse_2_3)
        out = out.view(out.size(0), -1)

        out = self.linear(out)
        out = self.drop(out)
        # with open('E:/ACS/weight1.txt', 'w') as f:
        #     print(w_up, file=f)
        # with open('E:/ACS/weight2.txt', 'w') as f:
        #     print(w_down, file=f)
        return out

def Acs_Res18_s(is_acs=False):
    return ResNet(Bottleneck, [3,4,4], num_classes=100, width_multiplier=1, is_acs=is_acs)
>>>>>>> 6229e39838423396b703141eb8cbb404f6c79090

# def Acs_Res50_l(is_acs=False):
#     return ResNet(Bottleneck, [4,5,5], num_classes=100, width_multiplier=1, is_acs=is_acs)
#
# def Acs_Res101(is_acs=False):
#     return ResNet(Bottleneck, [4,8,8], num_classes=100, width_multiplier=1, is_acs=is_acs)

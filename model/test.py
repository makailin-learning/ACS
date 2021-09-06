from model.utils import *
from model.utils import _fuse_bn_tensor
x=torch.rand([2,2,13,13])

#测试1x1的卷积核扩展为3x3的卷积核  分支1x1
def test_1to3(p):
    a0 = nn.Conv2d(2, 3, 1, 1, 0, bias=True)
    a1 = nn.Conv2d(2, 3, 3, 1, 1, bias=True)
    a1.weight.data = VI_multiscale(a0.weight, 3)*p
    a1.bias.data = a0.bias.data
    out1=a0(x*p)
    out2=a1(x)
    print(((out1 - out2) ** 2).sum())

# 分支3x3
def test_3(p):
    a0 = nn.Conv2d(2, 3, 3, 1, 1, bias=True)
    out1 = a0(x * p)
    a0.weight.data = a0.weight.data*p
    a0.bias.data = a0.bias.data
    out2=a0(x)
    print(((out1 - out2) ** 2).sum())

# 涉及到bn的都需要在model.eval()模式下来比较，不能直接比较, eval将bn的参数固定, 否则在模型转换时获得的参数会不一致
def test_bn(p):
    class model(nn.Module):
        def __init__(self,param=p):
            super().__init__()
            self.a0=nn.Conv2d(2,3,1,1,0,bias=False)
            self.a1=nn.BatchNorm2d(3)
            self.p=param
        def gek(self):
            k, b = I_fusebn(self.a0.weight, self.a1)
            return k,b
        def swith(self):
            k,b=self.gek()
            self.c = nn.Conv2d(2, 3, 1, 1, 0, bias=True)  # 过bn层的卷积偏差就没用了
            self.c.weight.data=k*self.p
            self.c.bias.data=b
            for para in self.parameters():
                para.detach_()
            self.__delattr__('a0')
            self.__delattr__('a1')
        def forward(self,x):
            if hasattr(self, 'c'):
                return self.c(x)
            return self.a1(self.a0(x))

    m=model()
    m.eval()
    out1 = m(x*p)
    m.swith()
    out2= m(x)
    print(((out1 - out2) ** 2).sum())

# 测试
def test_avg(p):
    # 池化和卷积都采用padding=1才能保证结果相同
    a0 = nn.AvgPool2d(kernel_size=3,stride=1,padding=1)
    a1=nn.Conv2d(2,3,3,1,1,bias=False)
    k = V_avg(channels=2, kernel_size=3, groups=1)
    a1.weight.data = k
    out1 = a0(x)
    out2 = a1(x)
    print(((out1 - out2) ** 2).sum())



def test_1x3_avg(p):
    class model(nn.Module):
        def __init__(self,param=p):
            super().__init__()
            self.a0=nn.Conv2d(2,3,1,1,0,bias=False)
            self.a1=nn.BatchNorm2d(3)
            self.avg=nn.AvgPool2d(kernel_size=3,stride=1,padding=1)
            self.p=param
        def gek(self):
            k0, b0 = I_fusebn(self.a0.weight, self.a1)
            k1= V_avg(channels=3, kernel_size=3, groups=1)
            b1=torch.zeros(k1.shape[0],device=k1.device)
            k,b=III_1x1_3x3(k0*self.p,b0,k1,b1)

            return k,b
        def swith(self):
            k,b=self.gek()
            self.c = nn.Conv2d(2, 3, 1, 1, 1, bias=True)
            self.c.weight.data=k
            self.c.bias.data=b
            for para in self.parameters():
                para.detach_()
            self.__delattr__('a0')
            self.__delattr__('a1')
            self.__delattr__('avg')
        def forward(self,x):
            if hasattr(self, 'c'):
                return self.c(x)
            return self.avg(self.a1(self.a0(x)))

    m=model()
    m.eval()
    out1 = m(x*p)
    m.swith()
    out2= m(x)
    print(((out1 - out2) ** 2).sum())

def test_1x3_3(p):
    class model(nn.Module):
        def __init__(self,param=p):
            super().__init__()
            self.a0=nn.Conv2d(2,3,1,1,0,bias=False)
            self.a1=nn.BatchNorm2d(3)
            self._3x3=nn.Conv2d(3,3,3,1,1,bias=True)
            self.p=param
        def gek(self):
            k0, b0 = I_fusebn(self.a0.weight, self.a1)
            k1= self._3x3.weight.data
            b1= self._3x3.bias.data
            k,b=III_1x1_3x3(k0*self.p,b0,k1,b1)
            return k,b
        def swith(self):
            k,b=self.gek()
            self.c = nn.Conv2d(2, 3, 1, 1, 1, bias=True)
            self.c.weight.data=k
            self.c.bias.data=b
            for para in self.parameters():
                para.detach_()
            self.__delattr__('a0')
            self.__delattr__('a1')
            self.__delattr__('_3x3')
        def forward(self,x):
            if hasattr(self, 'c'):
                return self.c(x)
            return self._3x3(self.a1(self.a0(x)))

    m=model()
    m.eval()
    out1 = m(x*p)
    m.swith()
    out2= m(x)
    print(((out1 - out2) ** 2).sum())

def test_3_1x3_3(p):
    class model(nn.Module):
        def __init__(self,param=p):
            super().__init__()
            self.a0=nn.Conv2d(2,3,1,1,0,bias=False)
            self.a1=nn.BatchNorm2d(3)
            self._3x3=nn.Conv2d(3,3,3,1,1,bias=True)
            self.main=nn.Conv2d(2,3,3,1,1,bias=True)
            self.p=param
        def gek(self):
            k0, b0 = I_fusebn(self.a0.weight, self.a1)
            k1= self._3x3.weight.data
            b1= self._3x3.bias.data
            k,b=III_1x1_3x3(k0*self.p,b0,k1,b1)
            k2,b2=self.main.weight.data*self.p,self.main.bias.data
            return IV_concat((k,k2),(b,b2))
        def swith(self):
            k,b=self.gek()
            self.c = nn.Conv2d(2, 3, 1, 1, 1, bias=True)
            self.c.weight.data=k
            self.c.bias.data=b
            for para in self.parameters():
                para.detach_()
            self.__delattr__('a0')
            self.__delattr__('a1')
            self.__delattr__('_3x3')
        def forward(self,x):
            if hasattr(self, 'c'):
                return self.c(x)
            out1=self._3x3(self.a1(self.a0(x)))
            out2=self.main(x*self.p)
            return torch.cat((out1,out2),dim=1)

    m=model()
    m.eval()
    out1 = m(x)
    m.swith()
    out2= m(x)
    print(((out1 - out2) ** 2).sum())

def test_3_1x3_avg(p):
    class model(nn.Module):
        def __init__(self,param=p):
            super().__init__()
            self.a0=nn.Conv2d(2,3,1,1,0,bias=False)
            self.a1=nn.BatchNorm2d(3)
            self.avg=nn.AvgPool2d(kernel_size=3,stride=1,padding=1)

            self.main=nn.Conv2d(2,3,3,1,1,bias=True)


            self.p=param
        def gek(self):
            k0, b0 = I_fusebn(self.a0.weight, self.a1)
            k1= V_avg(channels=3, kernel_size=3, groups=1)
            b1=torch.zeros(k1.shape[0],device=k1.device)
            k,b=III_1x1_3x3(k0*self.p,b0,k1,b1)
            k2=self.main.weight.data*self.p
            b2=self.main.bias.data

            return IV_concat((k,k2),(b,b2))
        def swith(self):
            k,b=self.gek()
            self.c = nn.Conv2d(2, 3, 1, 1, 1, bias=True)
            self.c.weight.data=k
            self.c.bias.data=b
            for para in self.parameters():
                para.detach_()
            self.__delattr__('a0')
            self.__delattr__('a1')
            self.__delattr__('avg')
        def forward(self,x):
            if hasattr(self, 'c'):
                return self.c(x)
            out1=self.avg(self.a1(self.a0(x)))
            out2=self.main(x*self.p)
            return torch.cat((out1,out2),dim=1)

    m=model()
    m.eval()
    out1 = m(x)
    m.swith()
    out2= m(x)
    print(((out1 - out2) ** 2).sum())

for i in range(5):
# test_1to3(0.36)
# test_3(0.62)
# test_bn(0.49)
# test_avg(0.21)
# test_1x3_avg(0.25)
# test_1x3_3(0.43)
# test_3_1x3_3(1.0)
    test_3_1x3_avg(1.0)

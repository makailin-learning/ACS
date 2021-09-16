import torch

from model.models import *

class tests(nn.Module):
    def __init__(self,test_id):
        super(tests, self).__init__()
        self.test_id=test_id

    def test_0(self,x):
        diff=[]
        device=x.device
        b,c,h,w=x.shape
        for i in range(5):
            m=ACS(in_channels=c,kernel_size=3,deploy=False,activation=nn.ReLU()).to(device)
            m.apply(weights_init)
            m.eval()
            train_y = m(x)
            m.switch_to_deploy()
            deploy_y = m(x)
            diff.append(((train_y - deploy_y) ** 2).sum())
        return diff

    def test_1(self,x):
        diff1 = []
        device=x.device
        m = Acs_Res50_s(is_acs=True).to(device)
        m.apply(weights_init)
        m.eval()
        train_y = m(x)
        for module in m.modules():
            if hasattr(module, 'switch_to_deploy'):
                module.switch_to_deploy()
        m.to(device)
        deploy_y = m(x)
        diff1.append(((train_y - deploy_y) ** 2).sum())
        return diff1

    def forward(self,x):
        if self.test_id==0:
            out=self.test_0(x)
        else:
            out=self.test_1(x)
        return out

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
x=torch.rand([32,3,224,224],device=device)
tests=tests(0)(x)
print(tests)

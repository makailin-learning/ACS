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
            #m=ConvBN(in_channels=64,kernel_size=3,deploy=False,activation=nn.ReLU())
            # 这段代码的用途？
            # for module in m.modules():
            #     if isinstance(module, torch.nn.BatchNorm2d):
            #         nn.init.uniform_(module.running_mean, 0, 0.1)
            #         nn.init.uniform_(module.running_var, 0, 0.1)
            #         nn.init.uniform_(module.weight, 0, 0.1)
            #         nn.init.uniform_(module.bias, 0, 0.1)

            m.eval()
            train_y = m(x)
            m.switch_to_deploy()
            deploy_y = m(x)
            diff.append(((train_y - deploy_y) ** 2).sum())
        return diff

    def test_1(self,x):
        diff1 = []
        device=x.device
        models_1 = create_Acs_Res50_s(is_acs=True).to(device)
        models_1.eval()
        train_y = models_1(x)
        for module in models_1.modules():
            if hasattr(module, 'switch_to_deploy'):
                module.switch_to_deploy()
        models_1.to(device)
        deploy_y = models_1(x)
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

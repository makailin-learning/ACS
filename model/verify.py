from model.models import *

def test_0():
    diff=[]
    for i in range(5):
        m=ACS(in_channels=64,kernel_size=3,deploy=False,activation=nn.ReLU())
        #m=ConvBN(in_channels=64,kernel_size=3,deploy=False,activation=nn.ReLU())
        # 这段代码的用途？
        # for module in m.modules():
        #     if isinstance(module, torch.nn.BatchNorm2d):
        #         nn.init.uniform_(module.running_mean, 0, 0.1)
        #         nn.init.uniform_(module.running_var, 0, 0.1)
        #         nn.init.uniform_(module.weight, 0, 0.1)
        #         nn.init.uniform_(module.bias, 0, 0.1)

        x=torch.rand([64,64,64,64])
        m.eval()
        train_y = m(x)
        m.switch_to_deploy()
        deploy_y = m(x)
        diff.append(((train_y - deploy_y) ** 2).sum())
    print('diff =',diff)

def test_1():
    diff = []
    x=torch.rand([2,3,416,416])
    models_0 = create_Res18()
    models_1 = create_Acs_Res18()
    models_1.eval()
    train_y = models_1(x)
    models_1.switch_to_deploy()
    deploy_y = models_1(x)
    diff.append(((train_y - deploy_y) ** 2).sum())
    print(train_y.shape,deploy_y.shape)
    print(diff)

test_1()

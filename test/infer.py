import argparse
from model.models import *
from utils.utils import test_preprocess
import torchvision.datasets as datasets
import torch.utils.data

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    trans = test_preprocess()
    test_dataset = datasets.CIFAR100(root=args.data, train=False, download=True, transform=trans)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)

    x = iter(test_loader)
    img, label = next(x)
    img=img.to(device)
    label=label.to(device)

    net=Acs_Res18_s(is_acs=True).to(device)
    checkpoint = torch.load(args.resume)
    net.load_state_dict(checkpoint['state_dict'])
    net.eval()

    out1=net(img)

    for module in net.modules():
        if hasattr(module, 'switch_to_deploy'):
            module.switch_to_deploy()

    out2=net(img)
    print('mkl',((out1 - out2) ** 2).sum())

    T_num=0
    F_num=0
    size=len(test_loader)
    print(size)
    for i, (images, target) in enumerate(test_loader):

        images = images.to(device)
        target = target.to(device)

        output = net(images)
        output=torch.sigmoid(output)
        output=torch.argmax(output)
        if target.data==output.data:
            T_num+=1
        else:
            F_num+=1
        # print('pred class:',output.data)
        # print('label class:',target.data)
    print(T_num,F_num)
    print(T_num/size)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='PyTorch ImageNet testing')
    parser.add_argument('--data', default='E:/cifar_100/cal', type=str, help='数据集路径')
    parser.add_argument('--resume', default='E:/ACS/ResNet-50_ACS_best.pth.tar', type=str, help='训练文件的路径')
    #parser.add_argument('--resume', default='E:/acs_model_store/18s_acs_y/ResNet-50_ACS_best.pth.tar', type=str, help='训练文件的路径')
    args=parser.parse_args()
    main(args)



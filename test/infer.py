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

    net=Acs_Res50_s(is_acs=False).to(device)
    checkpoint = torch.load(args.resume)
    net.load_state_dict(checkpoint['state_dict'])
    net.eval()
    for module in net.modules():
        if hasattr(module, 'switch_to_deploy'):
            module.switch_to_deploy()

    for i, (images, target) in enumerate(test_loader):

        images = images.to(device)
        target = target.to(device)

        output = net(images)
        output=torch.sigmoid(output)
        output=torch.argmax(output)
        print('pred class:',output.data)
        print('label class:',target.data)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='PyTorch ImageNet testing')
    parser.add_argument('--data', default='E:/cifar_100/cal', type=str, help='数据集路径')
    parser.add_argument('--resume', default='E:/ACS/ResNet-50_ACS_best.pth.tar', type=str, help='训练文件的路径')
    args=parser.parse_args()
    main(args)



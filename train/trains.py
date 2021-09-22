import sys
sys.path.append('E://ACS//')
# argparse 是 Python 内置的一个用于命令项选项与参数解析的模块
import argparse
# os 模块提供了非常丰富的方法用来处理文件和目录,os模块提供了  多数操作系统  的功能接口函数
import os
# random模块用于生成随机数
import random
# 高级的文件，文件夹，压缩包的处理模块,也主要用于文件的拷贝
import shutil
import time
import warnings

# 并行计算模块
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
# 是一个实现了各种优化算法的库
import torch.optim
# 用于在相同数据的不同进程中共享视图
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
# torchvision 库是服务于pytorch深度学习框架的,用来生成图片,视频数据集,和一些流行的模型类和预训练模型
# 它里面包含了很多数据集，而所有数据集都是 torch.utils.data.dataset 的子类
import torchvision.datasets as datasets
# 根据epoch训练次数来调整学习率（learning rate）的方法
from torch.optim.lr_scheduler import CosineAnnealingLR
from utils.utils import AverageMeter, accuracy, ProgressMeter, val_preprocess, strong_train_preprocess, standard_train_preprocess
from model.models import *

IMAGENET_TRAINSET_SIZE = 1281167   # imagenet-1K数据集的图片训练张数
CIFAR_TRAINSET_SIZE = 50048

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')

parser.add_argument('--data', default='E:/Image_Net/', type=str, help='数据集路径')
parser.add_argument('-a', '--arch', default='ResNet-50') # 使用 metavar 来指定一个替代名称
parser.add_argument('-t', '--blocktype', default='ACS', choices=['ACS', 'base'])
parser.add_argument('--epochs', default=120, type=int,help='训练世代')
parser.add_argument('--start-epoch', default=0, type=int, help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=128, type=int,help='训练批次数')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float, help='初始学习率', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, help='优化动量')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float, help='优化衰减率',dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int, help='打印频次')
parser.add_argument('--resume', default='', type=str, help='断点训练文件的路径')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',help='evaluate model on validation set')
parser.add_argument('--is_acs', action='store_true', default=True, help='是否采用acs模块')
parser.add_argument('--seed', default=7, type=int,help='为训练初始化随机种子')
parser.add_argument('--image_size', default=224, type=int,help='训练图像尺寸')
#parser.add_argument('--gpu', default=None, type=int,help='cuda设备使用id号')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def sgd_optimizer(net, lr, momentum, weight_decay):
    params = []
    for key, value in net.named_parameters():
        if not value.requires_grad:
            continue
        apply_lr = lr
        apply_wd = weight_decay
        if 'bias' in key:
            apply_lr = 2 * lr       #   Just a Caffe-style common practice. Made no difference.
        if 'depth' in key:
            apply_wd = 0
        print('set weight decay ', key, apply_wd)
        params += [{'params': [value], 'lr': apply_lr, 'weight_decay': apply_wd}]
    optimizer = torch.optim.SGD(params, lr, momentum=momentum)
    return optimizer

def main():
    args = parser.parse_args()

    # 在神经网络中，参数默认是进行随机初始化的。不同的初始化参数往往会导致不同的结果，当得到比较好的结果时我们通常希望这个结果是可以复现的
    # 在pytorch中，通过设置随机数种子也可以达到这么目的
    if args.seed is not None:
        random.seed(args.seed)
        # 在需要生成随机数的实验中，确保每次运行.py文件时，生成的随机数都是固定的，这样每次实验结果显示也就一致了
        torch.manual_seed(args.seed)  # 为CPU设置种子用于生成随机数
        # torch.cuda.manual_seed(args.seed)  # 为GPU设置种子用于生成随机数
        # 将这个 flag 置为True的话，每次返回的卷积算法将是确定的，即默认算法
        # 配合上设置 Torch 的随机种子为固定值的话,可以保证每次运行网络的时候相同输入的输出是固定的,做到每次训练结果可完好固定复现
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')
    net=Acs_Res18_s(is_acs=args.is_acs).to(device)

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss()

    optimizer = sgd_optimizer(net, args.lr, args.momentum, args.weight_decay)

    # T——max为一次学习率周期的迭代次数，即 T_max 个 epoch 之后重新设置学习率
    lr_scheduler = CosineAnnealingLR(optimizer=optimizer, T_max=args.epochs * CIFAR_TRAINSET_SIZE // args.batch_size)

    # 断点训练
    if args.resume:
        # 判断是否存在断点文件的有效路径
        if os.path.isfile(args.resume):
            # 一种格式化字符串的函数 str.format(),基本语法是通过 {} 和 : 来代替以前的%
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            # 获取断点处的世代数
            args.start_epoch = checkpoint['epoch']
            # 获取断点文件的准确率
            best_acc1 = checkpoint['best_acc1']
            if args.gpu is not None:
                # best_acc1 may be from a checkpoint from a different GPU
                best_acc1 = best_acc1.to(args.gpu)
            net.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['scheduler'])
            print("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    # 设置这个 flag 可以让内置的 cuDNN 的 auto-tuner 自动寻找最适合当前配置的高效算法，来达到优化运行效率的问题。
    cudnn.benchmark = True

    # Data loading code
    # 拼接路径 data_path+'train'
    traindir = os.path.join(args.data, 'train')
    valdir = os.path.join(args.data, 'val')

    # 数据读取及预处理模块
    trans = strong_train_preprocess(args.image_size) if 'ResNet' in args.arch else standard_train_preprocess(args.image_size)
    #print('aug is ', trans)

    """
    # ImageFolder是一个通用的数据加载器
    # root=traindir 图片存储的根目录，即各类别文件夹所在目录的上一级目录
    # transform=trans 对图片进行预处理的操作（函数）
    # target_transform=None 对图片类别进行预处理的操作,输入target,输出对其的转换.如果不传该参数,返回的顺序索引 0,1,2...
    
    返回的dataset都有以下三种属性: 
    self.classes 用一个 list 保存类别名称
    self.class_to_idx 类别对应的索引
    self.imgs 保存(img-path, class) tuple的 list  -->  [(img_data,class_id),(img_data,class_id),…]
    """
    # data shape: 32x32x3
    train_dataset = datasets.CIFAR100(root='E:/cifar_100/train', train=True, download=True,transform=trans)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    val_dataset = datasets.CIFAR100(root='E:/cifar_100/cal', train=False, download=True,transform=trans)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    """
    train_dataset = datasets.ImageFolder(traindir, trans)

    train_loader = torch.utils.data.DataLoader( train_dataset, batch_size=args.batch_size, shuffle=True,
                                                # pin_memory就是锁页内存,生成的Tensor数据最开始是属于内存中的锁页内存
                                                # 这样将内存的Tensor转义到GPU的显存就会更快一些,显卡中的显存全部是锁页内存
                                                pin_memory=True, drop_last=True)
    val_dataset = datasets.ImageFolder(valdir, val_preprocess(args.image_size))
    val_loader = torch.utils.data.DataLoader(val_dataset,batch_size=args.batch_size, shuffle=False, pin_memory=True)
    """
    if args.evaluate:
        validate(val_loader, net, criterion, args)
        return

    best_acc1 = 0
    for epoch in range(args.start_epoch, args.epochs):

        # adjust_learning_rate(optimizer, epoch, args)
        # train for one epoch
        train(train_loader, net, criterion, optimizer, epoch, args, lr_scheduler)
        print('\n')
        # evaluate on validation set
        acc1 = validate(val_loader, net, criterion, args)

        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)

        if is_best:
            torch.save(net,'E:/acs_model_store/best_model.pth')

        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': net.state_dict(),
            'best_acc1': best_acc1,
            'optimizer': optimizer.state_dict(),
            'scheduler': lr_scheduler.state_dict(),
        }, is_best, filename='{}_{}.pth.tar'.format( args.arch, args.blocktype))


def train(train_loader, model, criterion, optimizer, epoch, args, lr_scheduler):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    #lr = AverageMeter('now_lr',':6.6f')
    progress = ProgressMeter(len(train_loader),[batch_time, data_time, losses, top1, top5, lr_scheduler.get_lr()[0]],prefix="Epoch: [{}/{}]".format(epoch,args.epochs))

    # switch to train mode
    model.train()

    end = time.time()
    for i, (images, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        images = images.to(device)
        target = target.to(device)

        # compute output
        output = model(images)
        loss = criterion(output, target)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        lr_scheduler.step()

        if i % args.print_freq == 0:
            progress.display(i)
        #if i % 1000 == 0:
            #print('cur lr: ', lr_scheduler.get_lr()[0])

def validate(val_loader, model, criterion, args):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(len(val_loader),[batch_time, losses, top1, top5],prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):

            images = images.to(device)
            target = target.to(device)

            # compute output
            output = model(images)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)

        # TODO: this should also be done with the ProgressMeter
        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'.format(top1=top1, top5=top5))

    return top1.avg

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, filename.replace('.pth.tar', '_best.pth.tar'))

if __name__ == '__main__':
    main()
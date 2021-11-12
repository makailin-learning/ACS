import os
import time

import torch
import shutil
import tqdm
import cv2
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from PIL import Image

trans = transforms.Compose([
        # transforms.RandomResizedCrop(img_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])


def write_label():
    with open('E:/ImageNet/ILSVRC-2012/caffe_ilsvrc12/val.txt') as f:
        ff = f.readlines()
        for i in range(50000):
            label = ff[i].split(' ')[1]
            with open('E:/ImageNet/ILSVRC-2012/caffe_ilsvrc12/val_mkl.txt','a') as f1:
                f1.write(label)

# write_label()

train = datasets.ImageFolder('E:/ImageNet/train/ILSVRC2012_img_train/',trans)
# wjj = train.classes

# for i in wjj:
#     if os.path.exists('E:/ImageNet/val/'+i):
#         continue
#     else:
#         os.mkdir('E:/ImageNet/val/'+i)  # 创建文件夹目录

train_loader = DataLoader(train,batch_size=1,shuffle=False)
for i,data in enumerate(train_loader):
    img,label = data
    print(img.shape)
    print(label)
    input()
val = datasets.ImageFolder('E:/ImageNet/val/',trans)
val_loader = DataLoader(val,batch_size=1,shuffle=False)
for i,data in enumerate(val_loader):
    img,label = data
    print(img.shape)
    print(label)
    input()

def img_move(wjj):
    val_dir = os.listdir('E:/ImageNet/ILSVRC-2012/val')
    label_path = 'E:/ImageNet/ILSVRC-2012/caffe_ilsvrc12/val_mkl.txt'
    txt = open(label_path,'r')
    t = txt.readlines()
    tp=()
    val_list = []
    for i in tqdm.tqdm(range(50000)):
        label_id = int(t[i])
        wj_id = wjj[label_id]
        shutil.copyfile('E:/ImageNet/ILSVRC-2012/val/'+val_dir[i],'E:/ImageNet/val/'+wj_id+'/'+val_dir[i])
    print('move img ok')

import torch
import torchvision.transforms as transforms
import os
import datetime
from torch.utils.tensorboard import SummaryWriter

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

# 图像数据增强,resize/
def strong_train_preprocess(img_size):
    trans = transforms.Compose([
        # 随机大小、长宽比裁剪图片,随机裁剪面积比例，默认(0.08，1)
        # transforms.RandomResizedCrop(img_size),
        # 依据概率p对PIL图片进行水平翻转，p默认0.5
        transforms.RandomHorizontalFlip(),
        # 调整亮度、对比度、饱和度和色调 brightness,contrast,saturation,hue
        transforms.ColorJitter(brightness=0.4, saturation=0.4, hue=0.4),
        # 转为tensor，并归一化至[0-1]
        transforms.ToTensor(),
        # PCALighting(0.1, imagenet_pca['eigval'], imagenet_pca['eigvec']),
        # 对数据按通道进行标准化，即先减均值，再除以标准差
        normalize,
    ])
    print('---------------------- strong dataaug!')
    return trans

def standard_train_preprocess(img_size=224):
    trans = transforms.Compose([
        # transforms.RandomResizedCrop(img_size),
        transforms.ColorJitter(brightness=0.4, saturation=0.4, hue=0.4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])
    print('---------------------- weak dataaug!')
    return trans

def val_preprocess(img_size=224):
    trans = transforms.Compose([
        # transforms.Resize(img_size),
        # transforms.CenterCrop(),
        transforms.ToTensor(),
        normalize,
    ])
    return trans

def test_preprocess(img_size=224):
    trans = transforms.Compose([
        # transforms.Resize(img_size),
        # transforms.CenterCrop(),
        transforms.ToTensor(),
        normalize,
    ])
    return trans

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 生产tenorboard日志
class Logger(object):
    def __init__(self, log_dir, log_hist=True):
        if log_hist:
            log_dir = os.path.join(
                log_dir,
                datetime.datetime.now().strftime("%Y_%m_%d__%H_%M_%S"))
        self.writer = SummaryWriter(log_dir, comment="ACS")

    def scalar_summary(self, tag, value, step):
        self.writer.add_scalar(tag, value, step)

    def list_of_scalars_summary(self, tag_value_pairs, step):
        for tag, value in tag_value_pairs:
            self.writer.add_scalar(tag, value, step)

    def create_model(self, model, image_size):
        inputs = torch.randn((1, 3, image_size, image_size),device=device)
        self.writer.add_graph(model, input_to_model=inputs, verbose=False)
        preds = model(inputs)


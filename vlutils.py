import math, logging, shutil, torch
from os.path import join
import numpy as np
import torchvision.transforms as transforms
import torchvision.datasets as datasets

class Logger():
    def __init__(self, path="log.txt"):
        self.logger = logging.getLogger("Logger")
        self.file_handler = logging.FileHandler(path, "w")
        self.stdout_handler = logging.StreamHandler()
        self.logger.addHandler(self.file_handler)
        self.logger.addHandler(self.stdout_handler)
        self.stdout_handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(message)s'))
        self.file_handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(message)s'))
        self.logger.setLevel(logging.INFO)
    
    def info(self, txt):
        self.logger.info(txt)
    
    def close(self):
        self.file_handler.close()
        self.stdout_handler.close()

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
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

def save_checkpoint(state, is_best, path, filename="checkpoint.pth"):

    torch.save(state, join(path, filename))
    if is_best:
        shutil.copyfile(join(path, filename), join(path, 'model_best.pth'))

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def cifar10(path='data/cifar10', bs=100, num_workers=8):
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    train_dataset = datasets.CIFAR10(root=path, train=True, download=True,
                                                 transform=train_transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=bs, shuffle=True,
                                               num_workers=num_workers)

    test_dataset = datasets.CIFAR10(root=path, train=False, download=True,
                                                transform=test_transform)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=100, shuffle=False,
                                               num_workers=num_workers)

    return train_loader, test_loader

def cifar100(path='data/cifar100', bs=256, num_workers=8):
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    train_dataset = datasets.CIFAR100(root=path, train=True, download=True,
                                                 transform=train_transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=bs, shuffle=True,
                                               num_workers=num_workers)

    test_dataset = datasets.CIFAR100(root=path, train=False, download=True,
                                                transform=test_transform)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=100, shuffle=False, 
                                               num_workers=num_workers)

    return train_loader, test_loader

class CosAnnealingLR(object):
    def __init__(self, loader_len, epochs, lr_max, lr_min=0, warmup_epochs=0, last_epoch=-1):
        max_iters = loader_len * epochs
        warmup_iters = loader_len * warmup_epochs
        assert lr_max >= 0
        assert warmup_iters >= 0
        assert max_iters >= 0 and max_iters >= warmup_iters

        self.max_iters = max_iters
        self.lr_max = lr_max
        self.lr_min = lr_min
        self.warmup_iters = warmup_iters
        self.last_epoch = last_epoch

        assert self.last_epoch >= -1
        self.iter_counter = (self.last_epoch+1) * loader_len
        self.lr = 0 
    
    def restart(self, lr_max=None):
        if lr_max:
            self.lr_max = lr_max
        self.iter_counter = 0 

    def step(self):
        self.iter_counter += 1
        if self.warmup_iters > 0 and self.iter_counter <= self.warmup_iters:
            self.lr = float(self.iter_counter / self.warmup_iters) * self.lr_max
        else:
            self.lr = (1 + math.cos((self.iter_counter-self.warmup_iters) / \
                                    (self.max_iters - self.warmup_iters) * math.pi)) / 2 * self.lr_max
        return self.lr

class MultiStepLR(object):
    def __init__(self, loader_len, milestones, gamma=None, gammas=None, base_lr=0.1, warmup_epochs=0, last_epoch=-1):
        if gamma is not None and gammas is not None:
            raise ValueError("either specify gamma or gammas")
        if gamma is not None:
            gammas = [gamma] * len(milestones)
        assert isinstance(milestones, list)
        assert isinstance(gammas, list)
        assert len(milestones) == len(gammas)

        self.warmup_iters = warmup_epochs * loader_len
        self.loader_len = loader_len
        self.base_lr = base_lr
        self.lr = base_lr
        self.milestones = milestones
        self.gammas = gammas
        self.last_epoch = last_epoch

        assert self.last_epoch >= -1
        self.iter_counter = (self.last_epoch+1) * loader_len
        self.milestone_counter = 0

    def step(self):
        self.iter_counter += 1
        if self.warmup_iters > 0 and self.iter_counter <= self.warmup_iters:
            self.lr = float(self.iter_counter / self.warmup_iters) * self.base_lr
        else:
            if self.milestone_counter < len(self.milestones) and self.iter_counter == self.milestones[self.milestone_counter] * self.loader_len:
                self.lr = self.lr * self.gammas[self.milestone_counter]
                self.milestone_counter += 1
        return self.lr


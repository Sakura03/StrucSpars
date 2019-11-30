import torch, os, argparse, time, shutil
import torch.nn as nn
import numpy as np
from os.path import join, isfile, abspath
from vltools import Logger
from vltools.pytorch import save_checkpoint, AverageMeter, accuracy
from vltools.pytorch import datasets
from torch.optim.lr_scheduler import MultiStepLR
import resnet_cifar
from tensorboardX import SummaryWriter
from thop import profile

parser = argparse.ArgumentParser(description='PyTorch Cifar Training')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N', help='manual epoch number (useful on restarts)')
parser.add_argument('--print-freq', default=20, type=int, metavar='N', help='print frequency (default: 20)')
parser.add_argument('--data', metavar='DIR', default="/home/kai/.torch/data", help='path to dataset')
parser.add_argument('--dataset', default="cifar100", help='dataset')
parser.add_argument('--bs', '--batch-size', default=256, type=int, metavar='N', help='mini-batch size')
parser.add_argument('--epochs', default=160, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float, metavar='LR', help='initial learning rate')
parser.add_argument('--gamma', default=0.1, type=float, metavar='GM', help='decrease learning rate by gamma')
parser.add_argument('--milestones', default=[80, 120], type=eval, help='milestones for scheduling lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float, metavar='WD', help='weight decay')
parser.add_argument('--resume', default=None, type=str, metavar='PATH', help='path to latest checkpoint')
parser.add_argument('--tmp', default="results/tmp", type=str, help='tmp folder')
parser.add_argument('--randseed', type=int, default=None, help='random seed')
parser.add_argument('--fix-lr', action="store_true")
parser.add_argument('--depth', type=int, default=50, help='model depth')
args = parser.parse_args()

if args.randseed == None:
    args.randseed = np.random.randint(1000)
args.tmp = args.tmp.strip("/")
args.tmp = join(args.tmp, "resnet_cifar-depth%d"%args.depth, "baseline-seed%d" % args.randseed)

# Random seed
# According to https://pytorch.org/docs/master/notes/randomness.html
np.random.seed(args.randseed)
torch.manual_seed(args.randseed)
torch.cuda.manual_seed_all(args.randseed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

os.makedirs(args.tmp, exist_ok=True)

# loss function
criterion = torch.nn.CrossEntropyLoss()

tfboard_writer = SummaryWriter(log_dir=args.tmp)
logger = Logger(join(args.tmp, "log.txt"))

def main():
    logger.info(args)
    if args.dataset == "cifar10":
        train_loader, val_loader = datasets.cifar10(abspath(args.data), bs=args.bs)
        num_classes = 10
    elif args.dataset == "cifar100":
        train_loader, val_loader = datasets.cifar100(abspath(args.data), bs=args.bs)
        num_classes = 100

    # model and optimizer
    model_name = "resnet_cifar.resnet%d(num_classes=%d)" % (args.depth, num_classes)
    model = eval(model_name).cuda()
    model = nn.DataParallel(model)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    logger.info("Model details:")
    logger.info(model)
    logger.info("Optimizer details:")
    logger.info(optimizer)

    # records
    best_acc1 = 0

    # optionally resume from a checkpoint
    if args.resume is not None:
        if isfile(args.resume):
            shutil.copy(args.resume, args.tmp)
            logger.info("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            logger.info("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
        else:
            logger.info("=> no checkpoint found at '{}'".format(args.resume))

    scheduler = MultiStepLR(optimizer, milestones=args.milestones, gamma=args.gamma)
    
    for epoch in range(args.start_epoch, args.epochs):
        # train and evaluate
        loss = train(train_loader, model, optimizer, epoch)
        acc1, acc5 = validate(val_loader, model, epoch)
        if not args.fix_lr:
            scheduler.step()

        # remember best prec@1 and save checkpoint
        is_best = acc1 > best_acc1
        if is_best:
            best_acc1 = acc1
        save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_acc1': best_acc1,
                'optimizer' : optimizer.state_dict()
                }, is_best, path=args.tmp)

        logger.info("Best acc1=%.5f" % best_acc1)

        lr = optimizer.param_groups[0]["lr"]
        tfboard_writer.add_scalar('train/loss_epoch', loss, epoch)
        tfboard_writer.add_scalar('train/lr_epoch', lr, epoch)
        tfboard_writer.add_scalar('test/acc1_epoch', acc1, epoch)
        tfboard_writer.add_scalar('test/acc5_epoch', acc5, epoch)
        
    logger.info("Training done, ALL results saved to %s." % args.tmp)

def train(train_loader, model, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (data, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        target = target.cuda(non_blocking=True)

        output = model(data)
        loss = criterion(output, target)

        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), data.size(0))
        top1.update(acc1.item(), data.size(0))
        top5.update(acc5.item(), data.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # measure elapsed time
        batch_time.update(time.time() - end)
        lr = optimizer.param_groups[0]["lr"]

        if i % args.print_freq == 0:
            logger.info('Ep[{0}/{1}] It[{2}/{3}] Bt {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                        'Dt {data_time.val:.3f} ({data_time.avg:.3f}) Loss {loss.val:.3f} ({loss.avg:.3f}) '
                        'Prec@1 {top1.val:.3f} ({top1.avg:.3f}) Prec@5 {top5.val:.3f} ({top5.avg:.3f}) LR {lr:.3E}' \
                        .format(epoch, args.epochs, i, len(train_loader), batch_time=batch_time, data_time=data_time,
                                loss=losses, top1=top1, top5=top5, lr=lr))
        end = time.time()
    return losses.avg

def validate(val_loader, model, epoch):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (data, target) in enumerate(val_loader):
            target = target.cuda(non_blocking=True)
            data = data.cuda()
            # compute output
            output = model(data)
            loss = criterion(output, target)
            
            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), data.size(0))
            top1.update(acc1.item(), data.size(0))
            top5.update(acc5.item(), data.size(0))
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            if i % args.print_freq == 0:
                logger.info('Test: [{0}/{1}] Test Loss {loss.val:.3f} (avg={loss.avg:.3f}) Prec@1 {top1.val:.3f} '
                            '(avg={top1.avg:.3f}) Prec@5 {top5.val:.3f} (avg={top5.avg:.3f})' \
                            .format(i, len(val_loader), loss=losses, top1=top1, top5=top5))

        logger.info(' * Prec@1 {top1.avg:.5f} Prec@5 {top5.avg:.5f}'.format(top1=top1, top5=top5))
    return top1.avg, top5.avg

if __name__ == '__main__':
    main()

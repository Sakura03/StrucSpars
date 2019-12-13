import torch, os, argparse, time, utils
import torch.nn as nn
from torch.optim import lr_scheduler
import numpy as np
from scipy.io import savemat
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from os.path import join, isdir, isfile

from vltools import Logger
from vltools.pytorch import save_checkpoint, AverageMeter, accuracy
from models import VAMMNet, Mininet_pretrain, Fast_SCNN

try:
    import nvidia.dali.plugin.pytorch as plugin_pytorch
    from nvidia.dali.pipeline import Pipeline
    import nvidia.dali.ops as ops
    import nvidia.dali.types as types
except ImportError:
    raise ImportError("Please install DALI from https://www.github.com/NVIDIA/DALI to run this example.")

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
# arguments from command line
parser.add_argument('--config', default='configs/ilsvrc.yml', help='path to dataset')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N', help='manual epoch number (useful on restarts)')
parser.add_argument('--print-freq', default=50, type=int, metavar='N', help='print frequency (default: 50)')

# by default, arguments bellow will come from a config file
parser.add_argument('--model', metavar='STR', default=None, help='model')
parser.add_argument('--data', metavar='DIR', default=None, help='path to dataset')
parser.add_argument('--num_classes', default=None, type=int, metavar='N', help='Number of classes')
parser.add_argument('--bs', '--batch-size', default=None, type=int, metavar='N', help='mini-batch size')
parser.add_argument('--epochs', default=None, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('--lr', '--learning-rate', default=None, type=float, metavar='LR', help='initial learning rate')
parser.add_argument('--stepsize', '--step-size', default=None, type=int, metavar='SS', help='decrease learning rate every stepsize epochs')
parser.add_argument('--gamma', default=None, type=float, metavar='GM', help='decrease learning rate by gamma')
parser.add_argument('--momentum', default=None, type=float, metavar='M', help='momentum')
parser.add_argument('--weight-decay', '--wd', default=None, type=float, metavar='W', help='weight decay')                 
parser.add_argument('--resume', default=None, type=str, metavar='PATH', help='path to latest checkpoint')
parser.add_argument('--tmp', help='tmp folder', default=None)
parser.add_argument('--benchmark', dest='benchmark', action="store_true")
parser.add_argument('--gpu', default=None, type=int, metavar='N', help='GPU ID')

args = parser.parse_args()

assert isfile(args.config)
CONFIGS = utils.load_yaml(args.config)
CONFIGS = utils.merge_config(args, CONFIGS)

# Fix random seed for reproducibility
# According to https://pytorch.org/docs/master/notes/randomness.html
np.random.seed(CONFIGS["MISC"]["RAND_SEED"])
torch.manual_seed(CONFIGS["MISC"]["RAND_SEED"])
torch.cuda.manual_seed_all(CONFIGS["MISC"]["RAND_SEED"])
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

os.makedirs(CONFIGS["MISC"]["TMP"], exist_ok=True)

logger = Logger(join(CONFIGS["MISC"]["TMP"], "log.txt"))
logger.info("Training with DALI dataloader.")


# model and optimizer
model = CONFIGS["MODEL"]["MODEL"] + "(num_classes=%d)" % (CONFIGS["DATA"]["NUM_CLASSES"])
logger.info("Model: %s" % model)
model = eval(model)

if CONFIGS["CUDA"]["DATA_PARALLEL"]:
    logger.info("Model Data Parallel")
    model = nn.DataParallel(model).cuda()
else:
    model = model.cuda(device=CONFIGS["CUDA"]["GPU_ID"])

optimizer = torch.optim.SGD(
        model.parameters(),
        lr=CONFIGS["OPTIMIZER"]["LR"],
        momentum=CONFIGS["OPTIMIZER"]["MOMENTUM"],
        weight_decay=CONFIGS["OPTIMIZER"]["WEIGHT_DECAY"]
        )

logger.info("Model details:")
logger.info(model)
logger.info("Optimizer details:")
logger.info(optimizer)

"""
logger.info("Initial parameters details:")
for name, p in model.named_parameters():

    pmean, pstd = p.mean().item(), p.std().item()
    logger.info("%s, shape=%s, mean=%f, std=%f" % (name, str(p.shape), pmean, pstd))
"""

scheduler = lr_scheduler.MultiStepLR(
        optimizer,
        milestones=CONFIGS["OPTIMIZER"]["MILESTONES"],
        gamma=CONFIGS["OPTIMIZER"]["GAMMA"]
        )

# DALI data loader
class HybridTrainPipe(Pipeline):
    def __init__(self, batch_size, num_threads, device_id, data_dir, crop, num_gpus, dali_cpu=False):
        super(HybridTrainPipe, self).__init__(batch_size, num_threads, device_id, seed=12 + device_id)
        self.input = ops.MXNetReader(
                path=[data_dir+"/rec/train.rec"], 
                index_path=[data_dir+"/rec/train.idx"], 
                random_shuffle=True, 
                shard_id=device_id, 
                num_shards=num_gpus
                )
        
        if dali_cpu:
            dali_device = "cpu"
            self.decode = ops.HostDecoder(device=dali_device, output_type=types.RGB)
        else:
            dali_device = "gpu"
            # This padding sets the size of the internal nvJPEG buffers to be able to handle all images from full-sized ImageNet
            # without additional reallocations
            self.decode = ops.nvJPEGDecoder(device="mixed", output_type=types.RGB)

        self.rrc = ops.RandomResizedCrop(
                device=dali_device, 
                size=(crop, crop), 
                interp_type=types.INTERP_CUBIC, 
                random_area=[0.2, 1]
                )
        self.cmnp = ops.CropMirrorNormalize(
                device="gpu",
                output_dtype=types.FLOAT,
                output_layout=types.NCHW,
                crop=(crop, crop),
                image_type=types.RGB,
                mean=[0.485 * 255,0.456 * 255,0.406 * 255],
                std=[0.229 * 255,0.224 * 255,0.225 * 255]
                )
        self.coin = ops.CoinFlip(probability=0.5)
        print('DALI "{0}" variant'.format(dali_device))

    def define_graph(self):
        rng = self.coin()
        self.jpegs, self.labels = self.input(name="Reader")
        images = self.decode(self.jpegs)
        images = self.rrc(images)
        output = self.cmnp(images.gpu(), mirror=rng)
        return [output, self.labels]

class HybridValPipe(Pipeline):
    def __init__(self, batch_size, num_threads, device_id, data_dir, crop, size, num_gpus, dali_cpu=False):
        super(HybridValPipe, self).__init__(batch_size, num_threads, device_id, seed=12 + device_id)
        self.input = ops.MXNetReader(
                path = [data_dir+"/rec/val.rec"], 
                index_path=[data_dir+"/rec/val.idx"],
                random_shuffle=False, 
                shard_id=device_id, 
                num_shards=num_gpus
                )
                
        if dali_cpu:
            dali_device = "cpu"
            self.decode = ops.HostDecoder(device=dali_device, output_type=types.RGB)
        else:
            dali_device = "gpu"
            self.decode = ops.nvJPEGDecoder(device="mixed", output_type=types.RGB)
        self.res = ops.Resize(device=dali_device, resize_shorter=size, interp_type=types.INTERP_CUBIC)
        self.cmnp = ops.CropMirrorNormalize(
                device="gpu",
                output_dtype=types.FLOAT,
                output_layout=types.NCHW,
                crop=(crop, crop),
                image_type=types.RGB,
                mean=[0.485 * 255,0.456 * 255,0.406 * 255],
                std=[0.229 * 255,0.224 * 255,0.225 * 255]
                )

    def define_graph(self):
        self.jpegs, self.labels = self.input(name="Reader")
        images = self.decode(self.jpegs)
        images = self.res(images)
        output = self.cmnp(images)
        return [output, self.labels]

NUM_GPUS = 2
num_threads = 2
# train loader
pipes = [
        HybridTrainPipe(
                batch_size=int(CONFIGS["DATA"]["BS"] / NUM_GPUS), 
                num_threads=num_threads, 
                device_id=device_id, 
                data_dir=CONFIGS["DATA"]["DIR"], 
                crop=224, 
                num_gpus=NUM_GPUS, 
                dali_cpu=False
                ) for device_id in range(NUM_GPUS)
        ]
pipes[0].build()
train_loader = plugin_pytorch.DALIClassificationIterator(pipes, size=int(pipes[0].epoch_size("Reader")))

# val loader
pipes = [
        HybridValPipe(
                batch_size=int(100/NUM_GPUS), 
                num_threads=num_threads, 
                device_id=device_id, 
                data_dir=CONFIGS["DATA"]["DIR"], 
                crop=224, 
                size=256, 
                num_gpus=NUM_GPUS, 
                dali_cpu=False
                ) for device_id in range(NUM_GPUS)
        ]
pipes[0].build()
val_loader = plugin_pytorch.DALIClassificationIterator(pipes, size=int(pipes[0].epoch_size("Reader")))

# loss function
criterion = torch.nn.CrossEntropyLoss()

def main():
    logger.info(CONFIGS)

    # dataset
    assert isdir(CONFIGS["DATA"]["DIR"]), CONFIGS["DATA"]["DIR"]
    start_time = time.time()

    logger.info("Data loading done, %.3f sec elapsed." % (time.time() - start_time))

    # records
    best_acc1 = 0
    acc1_record = []
    acc5_record = []
    loss_record = []
    lr_record = []
    # optionally resume from a checkpoint
    if args.resume:
        if isfile(args.resume):
            logger.info("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            # records
            acc1_record = checkpoint["acc1_record"]
            acc5_record = checkpoint["acc5_record"]
            loss_record = checkpoint["loss_record"]
            lr_record = checkpoint["lr_record"]

            logger.info("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            logger.info("=> no checkpoint found at '{}'".format(args.resume))

    start_time = time.time()
    for epoch in range(args.start_epoch, CONFIGS["OPTIMIZER"]["EPOCHS"]):
        # train and evaluate
        loss = train(train_loader, epoch)
        acc1, acc5 = validate(val_loader)

        train_loader.reset()
        val_loader.reset()

        scheduler.step(epoch+1)

        # record stats
        loss_record.append(loss)
        acc1_record.append(acc1)
        acc5_record.append(acc5)
        lr_record.append(optimizer.param_groups[0]["lr"])

        # remember best prec@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1_record)
        best_acc5 = max(acc5_record)
        
        """
        # log parameters details
        logger.info("Epoch %d parameters details:" % epoch)
        for name, p in model.named_parameters():
            pmean, pstd = p.mean().item(), p.std().item()
            logger.info("%s, shape=%s, mean=%f, std=%f" % (name, str(p.shape), pmean, pstd))
        """
        
        logger.info("Best acc1=%.5f" % best_acc1)
        logger.info("Best acc5=%.5f" % best_acc5)

        # save checkpoint
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_acc1': best_acc1,
            'best_acc5': best_acc5,
            'optimizer' : optimizer.state_dict(),
            # records
            'acc1_record': acc1_record,
            'acc5_record': acc5_record,
            'lr_record': lr_record,
            'loss_record': loss_record,
            }, is_best, path=CONFIGS["MISC"]["TMP"])

        # We continously save records in case of interupt
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        axes[0].plot(acc1_record, color='r', linewidth=2)
        axes[0].plot(acc5_record, color='g', linewidth=2)
        axes[0].legend(['Top1 Accuracy (Best%.3f)' % max(acc1_record), 'Top5 Accuracy (Best%.3f)' % max(acc5_record)], loc="lower right")
        axes[0].grid(alpha=0.5, linestyle='dotted', linewidth=2, color='black')
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Precision")

        axes[1].plot(loss_record)
        axes[1].grid(alpha=0.5, linestyle='dotted', linewidth=2, color='black')
        axes[1].legend(["Loss"], loc="upper right")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Loss")
        
        axes[2].plot(lr_record)
        axes[2].grid(alpha=0.5, linestyle='dotted', linewidth=2, color='black')
        axes[2].legend(["Learning Rate"], loc="upper right")
        axes[2].set_xlabel("Epoch")
        axes[2].set_ylabel("Learning Rate")

        plt.tight_layout()
        plt.savefig(join(CONFIGS["MISC"]["TMP"], 'record.pdf'))
        plt.close(fig)

        record = dict({
                'acc1': np.array(acc1_record), 
                'acc5': np.array(acc5_record),
                'loss_record': np.array(loss_record), 
                "lr_record": np.array(lr_record)
                })

        savemat(join(CONFIGS["MISC"]["TMP"], 'record.mat'), record)

        t = time.time() - start_time                         # total seconds from starting
        elapsed = utils.DayHourMinute(t)
        t /= (epoch + 1) - args.start_epoch                  # seconds per epoch
        t = (CONFIGS["OPTIMIZER"]["EPOCHS"] - epoch - 1) * t # remaining seconds
        remaining = utils.DayHourMinute(t)

        logger.info("Epoch {0}/{1} finishied, auxiliaries saved to {2}. "
                    "Elapsed {elapsed.days:d} days {elapsed.hours:d} hours {elapsed.minutes:d} minutes. "
                    "Remaining {remaining.days:d} days {remaining.hours:d} hours {remaining.minutes:d} minutes.".format(
                    epoch, CONFIGS["OPTIMIZER"]["EPOCHS"], CONFIGS["MISC"]["TMP"], elapsed=elapsed, remaining=remaining))

    logger.info("Optimization done, ALL results saved to %s." % CONFIGS["MISC"]["TMP"])

def train(train_loader, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()

    train_loader_len = int(train_loader._size / CONFIGS["DATA"]["BS"])

    for i, data in enumerate(train_loader):
        # measure data loading time
        input = data[0]["data"]
        target = data[0]["label"].squeeze().cuda().long()

        data_time.update(time.time() - end)

        output = model(input)
        loss = criterion(output, target)
        
        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(acc1.item(), input.size(0))
        top5.update(acc5.item(), input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        lr = optimizer.param_groups[0]["lr"]
        if i % CONFIGS["MISC"]["LOGFREQ"] == 0:
            logger.info('Epoch[{0}/{1}] Iter[{2}/{3}] '
                  'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                  'Data Time {data_time.val:.3f} ({data_time.avg:.3f}) '
                  'Train Loss {loss.val:.3e} ({loss.avg:.3e}) '
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f}) '
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f}) '
                  'LR: {lr:.4f}'.format(
                   epoch, CONFIGS["OPTIMIZER"]["EPOCHS"], i, train_loader_len ,
                   batch_time=batch_time, data_time=data_time, loss=losses, top1=top1, top5=top5, lr=lr))
    return losses.avg

def validate(val_loader):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    # switch to evaluate mode
    model.eval()
    val_loader_len = int(val_loader._size / 100)
    with torch.no_grad():
        end = time.time()
        for i, data in enumerate(val_loader):
            
            input = data[0]["data"]
            target = data[0]["label"].squeeze().cuda().long()
            # compute output
            output = model(input)
            loss = criterion(output, target)
            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(acc1.item(), input.size(0))
            top5.update(acc5.item(), input.size(0))
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            if i % args.print_freq == 0:
                logger.info('Test: [{0}/{1}] '
                      'Test Loss {loss.val:.3e} (avg={loss.avg:.3e}) '
                      'Prec@1 {top1.val:.3f} (avg={top1.avg:.3f}) '
                      'Prec@5 {top5.val:.3f} (avg={top5.avg:.3f})'.format(
                       i, val_loader_len, loss=losses, top1=top1, top5=top5))

        logger.info(' * Prec@1 {top1.avg:.5f} Prec@5 {top5.avg:.5f}'
              .format(top1=top1, top5=top5))
    return top1.avg, top5.avg

if __name__ == '__main__':
    main()

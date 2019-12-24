import torch, os, argparse, time, warnings
import numpy as np
from PIL import Image
from os.path import join, isfile
from vlutils import Logger, save_checkpoint, AverageMeter, accuracy, MultiStepLR, CosAnnealingLR
from resnet_imagenet import GroupableConv2d
from utils import get_factors, get_sparsity, get_threshold
from utils import set_group_levels, mask_group, real_group
import resnet_imagenet
from tensorboardX import SummaryWriter
from thop import profile, count_hooks
# DALI data reader
from nvidia.dali.pipeline import Pipeline
import nvidia.dali.ops as ops
import nvidia.dali.types as types
from nvidia.dali.plugin.pytorch import DALIClassificationIterator
# distributed
import torch.distributed as dist
from apex.parallel import DistributedDataParallel as DDP
from apex.fp16_utils import FP16_Optimizer, BN_convert_float
warnings.filterwarnings("ignore", "(Possibly )?corrupt EXIF data", UserWarning)
warnings.simplefilter('error')

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N', help='manual epoch number (useful on restarts)')
parser.add_argument('--print-freq', default=50, type=int, metavar='N', help='print frequency (default: 50)')
parser.add_argument('-a', '--arch', default='resnet50', type=str, metavar='STR', help='model architecture')
parser.add_argument('--data', metavar='DIR', default="./data", help='path to dataset')
parser.add_argument('--num-classes', default=1000, type=int, metavar='N', help='Number of classes')
parser.add_argument('-j', '--workers', default=4, type=int, help='number of data loading workers')
parser.add_argument('--batch-size', default=64, type=int, metavar='N', help='mini-batch size')
parser.add_argument('--imsize', default=256, type=int, metavar='N', help='image resize size')
parser.add_argument('--imcrop', default=224, type=int, metavar='N', help='image crop size')
parser.add_argument('--epochs', default=110, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('--warmup', default=5, type=int, help="warmup epochs (small lr and do not impose sparsity)")
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float, metavar='LR', help='initial learning rate')
parser.add_argument('--gamma', default=0.1, type=float, metavar='GM', help='decrease learning rate by gamma')
parser.add_argument('--milestones', default=[30, 60, 90], type=eval, help='milestones for scheduling lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float, metavar='WD', help='weight decay')
parser.add_argument('--resume', default=None, type=str, metavar='PATH', help='path to the latest checkpoint')
parser.add_argument('--tmp', default="results/tmp", type=str, help='tmp folder')
# FP16
parser.add_argument('--fp16', action='store_true', help='Train the model with precision float16')
parser.add_argument('--finetune-fp16', action='store_true', help="Train the model with precision float16")
parser.add_argument('--static-loss-scale', type=float, default=1, help='Static loss scale, positive power of 2 values can improve fp16 convergence.')
parser.add_argument('--dynamic-loss-scale', action='store_true', help='Use dynamic loss scaling.  If supplied, this argument supersedes --static-loss-scale.')
# distributed
parser.add_argument("--local_rank", default=0, type=int)
parser.add_argument('--dali-cpu', action='store_true', help='Runs CPU based version of DALI pipeline.')
parser.add_argument('--use-rec', action="store_true", help='Use .rec data file')

parser.add_argument('--fix-lr', action="store_true", help='set true to fix learning rate')
parser.add_argument('--no-finetune', action="store_true", help='set true to disable finetuning')
parser.add_argument('--group1x1', action="store_true", help='set true to group conv1x1')
parser.add_argument('--sparsity', type=float, default=1e-5, help='sparsity regularization')
parser.add_argument('--sparse-thres', type=float, default=0.1, help='sparse threshold')
parser.add_argument('--finetune-lr', type=int, default=1e-1, help="finetune lr")
parser.add_argument('--finetune-epochs', type=int, default=120, help="finetune epochs")
parser.add_argument('--finetune-milestones', type=eval, default=[30, 60, 90], help="finetune milestones")
parser.add_argument('--finetune-weight-decay', type=float, default=1e-4, help="finetune weight decay")
parser.add_argument('--init-iters', type=int, default=50, help='Initial iterations')
parser.add_argument('--epoch-iters', type=int, default=20, help='Iterations for each epoch')
parser.add_argument('--iter-iters', type=int, default=5, help='Iterations for each 500 training iterations')
parser.add_argument('--power', type=float, default=0.3, help='Decay rate in the penalty matrix')
parser.add_argument('--percent', type=float, default=0.5, help='remaining parameter percent')
args = parser.parse_args()

# DALI pipelines
class HybridTrainPipe(Pipeline):
    def __init__(self, batch_size, num_threads, device_id, data_dir, crop, dali_cpu=False):
        super(HybridTrainPipe, self).__init__(batch_size, num_threads, device_id, seed=12 + device_id)
        if args.use_rec:
            # MXnet rec reader
            self.input = ops.MXNetReader(path=join(data_dir, "train.rec"), index_path=join(data_dir, "train.idx"),
                                        random_shuffle=True, shard_id=args.local_rank, num_shards=args.world_size)
        else:
            # image reader
            self.input = ops.FileReader(file_root=data_dir, shard_id=args.local_rank, num_shards=args.world_size, random_shuffle=True)
        #let user decide which pipeline works him bets for RN version he runs
        dali_device = 'cpu' if dali_cpu else 'gpu'
        decoder_device = 'cpu' if dali_cpu else 'mixed'
        # This padding sets the size of the internal nvJPEG buffers to be able to handle all images from full-sized ImageNet
        # without additional reallocations
        device_memory_padding = 211025920 if decoder_device == 'mixed' else 0
        host_memory_padding = 140544512 if decoder_device == 'mixed' else 0
        self.decode = ops.ImageDecoderRandomCrop(device=decoder_device, output_type=types.RGB,
                                                 device_memory_padding=device_memory_padding,
                                                 host_memory_padding=host_memory_padding,
                                                 random_aspect_ratio=[0.8, 1.25],
                                                 random_area=[0.1, 1.0],
                                                 num_attempts=100)
        self.res = ops.Resize(device=dali_device, resize_x=crop, resize_y=crop, interp_type=types.INTERP_TRIANGULAR)
        self.cmnp = ops.CropMirrorNormalize(device="gpu",
                                            output_dtype=types.FLOAT,
                                            output_layout=types.NCHW,
                                            crop=(crop, crop),
                                            image_type=types.RGB,
                                            mean=[0.485 * 255,0.456 * 255,0.406 * 255],
                                            std=[0.229 * 255,0.224 * 255,0.225 * 255])
        self.coin = ops.CoinFlip(probability=0.5)
        print('DALI "{0}" variant'.format(dali_device))

    def define_graph(self):
        rng = self.coin()
        self.jpegs, self.labels = self.input(name="Reader")
        images = self.decode(self.jpegs)
        images = self.res(images)
        output = self.cmnp(images.gpu(), mirror=rng)
        return [output, self.labels]

class HybridValPipe(Pipeline):
    def __init__(self, batch_size, num_threads, device_id, data_dir, crop, size):
        super(HybridValPipe, self).__init__(batch_size, num_threads, device_id, seed=12 + device_id)
        if args.use_rec:
            self.input = ops.MXNetReader(path=join(data_dir, "val.rec"), index_path=join(data_dir, "val.idx"),
                                     random_shuffle=False, shard_id=args.local_rank, num_shards=args.world_size)
        else:
            self.input = ops.FileReader(file_root=data_dir, shard_id=args.local_rank, num_shards=args.world_size, random_shuffle=False)
        self.decode = ops.ImageDecoder(device="mixed", output_type=types.RGB)
        self.res = ops.Resize(device="gpu", resize_shorter=size, interp_type=types.INTERP_TRIANGULAR)
        self.cmnp = ops.CropMirrorNormalize(device="gpu",
                                            output_dtype=types.FLOAT,
                                            output_layout=types.NCHW,
                                            crop=(crop, crop),
                                            image_type=types.RGB,
                                            mean=[0.485 * 255,0.456 * 255,0.406 * 255],
                                            std=[0.229 * 255,0.224 * 255,0.225 * 255])
    def define_graph(self):
        self.jpegs, self.labels = self.input(name="Reader")
        images = self.decode(self.jpegs)
        images = self.res(images)
        output = self.cmnp(images)
        return [output, self.labels]  

torch.backends.cudnn.benchmark = True
os.makedirs(args.tmp, exist_ok=True)

# loss function
criterion = torch.nn.CrossEntropyLoss()
custom_ops = {GroupableConv2d: count_hooks.count_convNd}

if args.local_rank == 0:
    tfboard_writer = SummaryWriter(log_dir=args.tmp)
    logger = Logger(join(args.tmp, "log.txt"))

args.distributed = False
if 'WORLD_SIZE' in os.environ:
    args.distributed = int(os.environ['WORLD_SIZE']) > 1

if args.fp16 or args.finetune_fp16:
    assert torch.backends.cudnn.enabled, "fp16 mode requires cudnn backend to be enabled."

def main():
    if args.local_rank == 0:
        logger.info(args)

    if args.distributed:
        args.gpu = args.local_rank % torch.cuda.device_count()
        torch.cuda.set_device(args.gpu)
        torch.distributed.init_process_group(backend='nccl', init_method='env://')
        args.world_size = torch.distributed.get_world_size()
        
    if args.use_rec:
        traindir = args.data
        valdir = args.data
    else:
        traindir = join(args.data, "train")
        valdir = join(args.data, "val")
    
    pipe = HybridTrainPipe(batch_size=args.batch_size, num_threads=args.workers,
                           device_id=args.local_rank, data_dir=traindir, 
                           crop=args.imcrop, dali_cpu=args.dali_cpu)
    pipe.build()
    train_loader = DALIClassificationIterator(pipe, size=int(pipe.epoch_size("Reader") / args.world_size))

    pipe = HybridValPipe(batch_size=args.batch_size, num_threads=args.workers,
                         device_id=args.local_rank, data_dir=valdir,
                         crop=args.imcrop, size=args.imsize)
    pipe.build()
    val_loader = DALIClassificationIterator(pipe, size=int(pipe.epoch_size("Reader") / args.world_size))
    train_loader_len = int(train_loader._size / args.batch_size)
    
    # model and optimizer
    group1x1 = "True" if args.group1x1 else "False"
    model_name = "resnet_imagenet.%s(num_classes=%d, group1x1=%s, power=%f)" % (args.arch, args.num_classes, group1x1, args.power)
    model = eval(model_name).cuda()
    if args.local_rank == 0:
        logger.info("Model details:")
        logger.info(model)

    if args.finetune_fp16:
        model = BN_convert_float(model.half())
    model = DDP(model, delay_allreduce=False)

    optimizer_finetune = torch.optim.SGD(model.parameters(), lr=args.finetune_lr, momentum=args.momentum, weight_decay=args.finetune_weight_decay)
    if args.local_rank == 0:
        logger.info("Optimizer details:")
        logger.info(optimizer_finetune)
    if args.finetune_fp16:
        optimizer_finetune = FP16_Optimizer(optimizer_finetune, static_loss_scale=args.static_loss_scale,
                                            dynamic_loss_scale=args.dynamic_loss_scale, verbose=False)
    
    scheduler_finetune = CosAnnealingLR(loader_len=train_loader_len, epochs=args.finetune_epochs,
                                        lr_max=args.finetune_lr, warmup_epochs=args.warmup)
    
    # optionally resume from a checkpoint
    if args.resume:
        if isfile(args.resume):
            if args.local_rank == 0:
                logger.info("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume, map_location=torch.device('cpu'))
            model.load_state_dict(checkpoint['state_dict'], strict=False)
            if args.local_rank == 0:
                logger.info("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
        else:
            if args.local_rank == 0:
                logger.info("=> no checkpoint found at '{}'".format(args.resume))

    if args.local_rank == 0:
        # evaluate before grouping
        logger.info("evaluating before grouping...")
    acc1, acc5 = validate(val_loader, model, args.epochs)
    if args.local_rank == 0:
        tfboard_writer.add_scalar('finetune/acc1-epoch', acc1, global_step=-2)
        tfboard_writer.add_scalar('finetune/acc5-epoch', acc5, global_step=-2)

    # mask grouping
    # thres = get_threshold(model.module, args.percent)
    thres = args.sparse_thres
    # if args.local_rank == 0:
    #     logger.info("Prune rate %.3e, threshold %.3e" % (args.percent, thres))
    # group_levels = mask_group(model.module, get_factors(model.module), thres, logger=logger if args.local_rank == 0 else None)
    fake_mask_group(model.module, logger=logger if args.local_rank==0 else None)

    if args.local_rank == 0:
        logger.info("evaluating after grouping...")
    acc1, acc5 = validate(val_loader, model, args.epochs)

    # real grouping
    # real_group(model.module, group_levels)
    # if args.local_rank == 0:
    #     flops, params = profile(model.module, inputs=(torch.randn(1, 3, 32, 32).cuda(),), custom_ops=custom_ops, verbose=False)
    #     logger.info("FLOPs %.3e, Params %.3e (after real grouping)" % (flops, params))

    #     logger.info("evaluating after real grouping...")
    # acc1, acc5 = validate(val_loader, model, args.epochs)

    if args.local_rank == 0:
        tfboard_writer.add_scalar('finetune/acc1-epoch', acc1, global_step=-1)
        tfboard_writer.add_scalar('finetune/acc5-epoch', acc5, global_step=-1)

    if args.local_rank == 0:
        best_acc1 = 0    
    for epoch in range(0, args.finetune_epochs):
        # train and evaluate
        loss = train(train_loader, model, optimizer_finetune, scheduler_finetune, epoch)
        acc1, acc5 = validate(val_loader, model, epoch, finetune=True)
        
        if args.local_rank == 0:
            # remember best prec@1 and save checkpoint
            is_best = acc1 > best_acc1
            if is_best:
                best_acc1 = acc1
    
            tfboard_writer.add_scalar('finetune/loss-epoch', loss, epoch)
            tfboard_writer.add_scalar('finetune/acc1-epoch', acc1, epoch)
            tfboard_writer.add_scalar('finetune/acc5-epoch', acc5, epoch)
            logger.info("Best acc1=%.5f" % best_acc1)

            # save checkpoint
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_acc1': best_acc1,
                'optimizer' : optimizer_finetune.state_dict()
                }, is_best, path=args.tmp, filename="checkpoint-retrain.pth")

def train(train_loader, model, optimizer, scheduler, epoch):
    if args.local_rank == 0:
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()

    train_loader_len = int(np.ceil(train_loader._size/args.batch_size))
    # switch to train mode
    model.train()
    
    if args.local_rank == 0:
        end = time.time()
    for i, data in enumerate(train_loader):
        target = data[0]["label"].squeeze().cuda().long()
        data = data[0]["data"]
        if args.finetune_fp16:
            data = data.half()

        # measure data loading time
        if args.local_rank == 0:
            data_time.update(time.time() - end)
        
        output = model(data)
        loss = criterion(output, target)

        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        reduced_loss = reduce_tensor(loss)
        acc1 = reduce_tensor(acc1)
        acc5 = reduce_tensor(acc5)
        
        if args.local_rank == 0:
            losses.update(reduced_loss.item(), data.size(0))
            top1.update(acc1.item(), data.size(0))
            top5.update(acc5.item(), data.size(0))

        # compute and adjust lr
        if scheduler is not None:
            lr = scheduler.step()
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
        
        # compute gradient and do SGD step
        optimizer.zero_grad()
        if args.finetune_fp16:
            optimizer.backward(loss)
        else:
            loss.backward()
        optimizer.step()
        
        # torch.cuda.synchronize()
        # measure elapsed time
        if args.local_rank == 0:
            batch_time.update(time.time() - end)
            lr = optimizer.param_groups[0]["lr"]
        
        if i % args.print_freq == 0 and args.local_rank == 0:
            tfboard_writer.add_scalar("finetune/iter-lr", lr, epoch*train_loader_len+i)
            logger.info('Ep[{0}/{1}] It[{2}/{3}] Bt {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                        'Dt {data_time.val:.3f} ({data_time.avg:.3f}) Loss {loss.val:.3f} ({loss.avg:.3f}) '
                        'Prec@1 {top1.val:.3f} ({top1.avg:.3f}) Prec@5 {top5.val:.3f} ({top5.avg:.3f}) LR {lr:.3E}' \
                        .format(epoch, args.finetune_epochs, i, train_loader_len, batch_time=batch_time, data_time=data_time,
                                loss=losses, top1=top1, top5=top5, lr=lr))
        
        if args.local_rank == 0:
            end = time.time()
        
    train_loader.reset()
    loss = torch.tensor([losses.avg]).cuda() if args.local_rank == 0 else torch.tensor([0.]).cuda()
    dist.broadcast(loss, 0)
    return loss.cpu().item()

@torch.no_grad()
def validate(val_loader, model, epoch, finetune=False):
    losses = AverageMeter()
    if args.local_rank == 0:
        batch_time = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()
    # switch to evaluate mode
    model.eval()
    if args.local_rank == 0:
        val_loader_len = int(np.ceil(val_loader._size/args.batch_size))
        
        end = time.time()
    for i, data in enumerate(val_loader):
        target = data[0]["label"].squeeze().cuda().long()
        data = data[0]["data"]
        if (finetune and args.finetune_fp16) or ((not finetune) and args.fp16):
            data = data.half()
        # compute output
        output = model(data)
        loss = criterion(output, target)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        if args.distributed:
            reduced_loss = reduce_tensor(loss.data)
            acc1 = reduce_tensor(acc1)
            acc5 = reduce_tensor(acc5)
        else:
            reduced_loss = loss.data

        if args.local_rank == 0:
            losses.update(reduced_loss.item(), data.size(0))
            top1.update(acc1.item(), data.size(0))
            top5.update(acc5.item(), data.size(0))
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                logger.info('Test: [{0}/{1}] Test Loss {loss.val:.3f} (avg={loss.avg:.3f}) Prec@1 {top1.val:.3f} '
                            '(avg={top1.avg:.3f}) Prec@5 {top5.val:.3f} (avg={top5.avg:.3f})' \
                            .format(i, val_loader_len, loss=losses, top1=top1, top5=top5))

    if args.local_rank == 0:
        logger.info(' * Prec@1 {top1.avg:.5f} Prec@5 {top5.avg:.5f}'.format(top1=top1, top5=top5))

    val_loader.reset()
    top1 = torch.tensor([top1.avg]).cuda() if args.local_rank == 0 else torch.tensor([0.]).cuda()
    top5 = torch.tensor([top5.avg]).cuda() if args.local_rank == 0 else torch.tensor([0.]).cuda()
    dist.broadcast(top1, 0)
    dist.broadcast(top5, 0)
    return top1.cpu().item(), top5.cpu().item()

def reduce_tensor(tensor):
    rt = tensor.clone()
    dist.reduce(rt, 0, op=dist.ReduceOp.SUM)
    rt /= args.world_size
    return rt

def fake_mask_group(model, logger):
    for name, m in model.named_modules():
        if isinstance(m, GroupableConv2d):
            unique_penalties = m.shuffled_penalty.unique()
            m.group_level = len(unique_penalties) - 1
            m.mask_grouped = True
            m.permuted_mask = (m.shuffled_penalty <= m.shuffled_penalty.unique(sorted=True)[1]).to(m.weight.dtype)
            assert torch.sum(m.permuted_mask) == m.in_channels * m.out_channels / (2 ** (m.group_level - 1))
            m.weight.data *= m.permuted_mask
            if logger is not None:
                logger.info("Layer %s, weight size %s, group level %d" % (name, str(list(m.weight.size())), m.group_level))

if __name__ == '__main__':
    main()

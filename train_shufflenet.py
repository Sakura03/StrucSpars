import torch, os, argparse, time, warnings
import torch.nn as nn
from os.path import join, isfile
from utils import DataIterator, CrossEntropyLabelSmooth, get_parameters
from vlutils import Logger, save_checkpoint, AverageMeter, accuracy, LinearLR
from model import *
from tensorboardX import SummaryWriter
from thop import profile
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

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training of ShuffleNetV1')
parser.add_argument('--start-iter', default=0, type=int, metavar='N', help='manual iteration number (useful on restarts)')
parser.add_argument('--print-freq', default=50, type=int, metavar='N', help='print frequency (default: 50)')
parser.add_argument('--val-freq', default=10000, type=int, metavar='N', help='validation frequency (default: 10000)')
parser.add_argument('-a', '--arch', default='shufflenet_v1', type=str, metavar='STR', help='model architecture')
parser.add_argument('--group', type=int, default=3, metavar='N', help='group number')
parser.add_argument('--model-size', type=str, default='1.0x', choices=['0.5x', '1.0x', '1.5x', '2.0x'], help='size of the model')
parser.add_argument('--data', metavar='DIR', default="./data", help='path to dataset')
parser.add_argument('--num-classes', default=1000, type=int, metavar='N', help='Number of classes')
parser.add_argument('-j', '--workers', default=8, type=int, help='number of data loading workers')
parser.add_argument('--batch-size', default=256, type=int, metavar='N', help='mini-batch size')
parser.add_argument('--imsize', default=256, type=int, metavar='N', help='image resize size')
parser.add_argument('--imcrop', default=224, type=int, metavar='N', help='image crop size')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
parser.add_argument('--label-smooth', type=float, default=0.1, help='label smoothing')
parser.add_argument('--resume', default=None, type=str, metavar='PATH', help='path to the latest checkpoint')
parser.add_argument('--tmp', default="results/tmp", type=str, help='tmp folder')
# FP16
parser.add_argument('--fp16', action='store_true', help="Train the model with precision float16")
parser.add_argument('--static-loss-scale', type=float, default=1, help='Static loss scale, positive power of 2 values can improve fp16 convergence.')
parser.add_argument('--dynamic-loss-scale', action='store_true', help='Use dynamic loss scaling.  If supplied, this argument supersedes --static-loss-scale.')
# distributed
parser.add_argument("--local_rank", default=0, type=int)
parser.add_argument('--dali-cpu', action='store_true', help='Runs CPU based version of DALI pipeline.')
parser.add_argument('--use-rec', action="store_true", help='Use .rec data file')

parser.add_argument('--lr', '--learning-rate', type=float, default=0.5, help=" init learning rate")
parser.add_argument('--total-iters', type=int, default=300000, help="total iterations")
parser.add_argument('--wd', '--weight-decay', type=float, default=4e-5, help="weight decay")
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
        self.twist = ops.ColorTwist(device=dali_device)
        self.saturation = ops.Uniform(range=[0.6, 1.4])
        self.contrast = ops.Uniform(range=[0.6, 1.4])
        self.brightness = ops.Uniform(range=[0.6, 1.4])
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
        images = self.twist(images, saturation=self.saturation(), contrast=self.contrast(), brightness=self.brightness())
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
criterion_smooth = CrossEntropyLabelSmooth(1000, args.label_smooth).cuda()

if args.local_rank == 0:
    tfboard_writer = SummaryWriter(log_dir=args.tmp)
    logger = Logger(join(args.tmp, "log.txt"))

args.distributed = False
if 'WORLD_SIZE' in os.environ:
    args.distributed = int(os.environ['WORLD_SIZE']) > 1

if args.fp16:
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
    train_dataprovider = DataIterator(train_loader)
    print(int(train_dataprovider.dataloader._size/args.batch_size)) ###TODO: tmp

    pipe = HybridValPipe(batch_size=50, num_threads=args.workers,
                         device_id=args.local_rank, data_dir=valdir,
                         crop=args.imcrop, size=args.imsize)
    pipe.build()
    val_loader = DALIClassificationIterator(pipe, size=int(pipe.epoch_size("Reader") / args.world_size))
    # val_dataprovider = DataIterator(val_loader)
    
    # model and optimizer
    model_name = "%s(num_classes=%d, groupable=False, model_size=%s, group=%d)" % \
                 (args.arch, args.num_classes, args.model_size, args.group)
    model = eval(model_name).cuda()
    if args.local_rank == 0:
        logger.info("Model details:")
        logger.info(model)
    if args.fp16:
        model = BN_convert_float(model.half())
    model = DDP(model, delay_allreduce=False)

    optimizer = torch.optim.SGD(get_parameters(model), lr=args.lr, momentum=args.momentum, weight_decay=args.wd)
    if args.local_rank == 0:
        logger.info("Optimizer details:")
        logger.info(optimizer)
    if args.fp16:
        optimizer = FP16_Optimizer(optimizer, static_loss_scale=args.static_loss_scale,
                                   dynamic_loss_scale=args.dynamic_loss_scale, verbose=False)

    # resume from a checkpoint
    if args.resume is not None:
        assert isfile(args.resume), "=> no checkpoint found at '{}'".format(args.resume)
        if args.local_rank == 0:
            logger.info("=> loading checkpoint '{}'".format(args.resume))
        checkpoint = torch.load(args.resume, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint['state_dict'], strict=False)
        optimizer.load_state_dict(checkpoint['optimizer'])
        args.start_iter = checkpoint['iter']
        if args.local_rank == 0: 
            best_acc1 = checkpoint['best_acc1']
            logger.info("=> loaded checkpoint '{}' (iter {})".format(args.resume, checkpoint['iter']))    
    else:
        args.start_iter = 0
        if args.local_rank == 0: best_acc1 = 0    
    
    scheduler = LinearLR(total_iters=args.total_iters, lr_max=args.lr, last_iter=args.start_iter-1)
    
    flops, params = profile(model.module, inputs=(torch.randn(1, 3, 224, 224).cuda(),), verbose=False)
    if args.local_rank == 0:
        logger.info("Baseline model: %.3e FLOPs, %.3e params" % (flops, params))

    all_iters = args.start_iter
    while all_iters < args.total_iters:
        loss, all_iters = train(train_dataprovider, model, optimizer, scheduler, bn_process=False, all_iters=all_iters)
        acc1, acc5 = validate(val_loader, model, all_iters=all_iters)
        
        if args.local_rank == 0:
            # remember best prec@1 and save checkpoint
            is_best = acc1 > best_acc1
            if is_best:
                best_acc1 = acc1
                
            # save checkpoint
            save_checkpoint({
                'iter': all_iters + 1,
                'state_dict': model.state_dict(),
                'best_acc1': best_acc1,
                'optimizer' : optimizer.state_dict()
                }, is_best, path=args.tmp, filename="checkpoint.pth")
            # log info into tfboard
            tfboard_writer.add_scalar('train/loss-iter', loss, all_iters)
            tfboard_writer.add_scalar('train/acc1-iter', acc1, all_iters)
            tfboard_writer.add_scalar('train/acc5-iter', acc5, all_iters)
            logger.info("Best acc1=%.5f" % best_acc1)
    
    loss, _ = train(train_dataprovider, model, optimizer, scheduler, bn_process=False, all_iters=all_iters)
    acc1, acc5 = validate(val_loader, model, all_iters=all_iters)
    
    if args.local_rank == 0:
        # remember best prec@1 and save checkpoint
        is_best = acc1 > best_acc1
        if is_best:
            best_acc1 = acc1
    # save checkpoint
    save_checkpoint({
        'iter': all_iters + 1,
        'state_dict': model.state_dict(),
        'best_acc1': best_acc1,
        'optimizer' : optimizer.state_dict()
        }, is_best, path=args.tmp, filename="checkpoint-final.pth")
        
def adjust_bn_momentum(model, iters):
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.momentum = 1 / iters

def train(train_dataprovider, model, optimizer, scheduler, bn_process=False, all_iters=None):
    if args.local_rank == 0:
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()

    # switch to train mode
    model.train()
    
    if args.local_rank == 0:
        end = time.time()
    curr_iters = int(train_dataprovider.dataloader._size/args.batch_size) if bn_process else args.val_freq
    for iters in enumerate(curr_iters):
        all_iters += 1
        # compute and adjust lr
        lr = scheduler.step()
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        
        # optionally adjust BN momentum
        if bn_process:
            adjust_bn_momentum(model, iters+1)

        data, target = train_dataprovider.next()
        if args.fp16:
            data = data.half()

        # measure data loading time
        if args.local_rank == 0:
            data_time.update(time.time() - end)
        
        output = model(data)
        loss = criterion_smooth(output, target)

        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        reduced_loss = reduce_tensor(loss)
        acc1 = reduce_tensor(acc1)
        acc5 = reduce_tensor(acc5)
        
        if args.local_rank == 0:
            losses.update(reduced_loss.item(), data.size(0))
            top1.update(acc1.item(), data.size(0))
            top5.update(acc5.item(), data.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        if args.fp16:
            optimizer.backward(loss)
        else:
            loss.backward()
        optimizer.step()
        
        # torch.cuda.synchronize()
        # measure elapsed time
        if args.local_rank == 0:
            batch_time.update(time.time() - end)
        
        if all_iters % args.print_freq == 0 and args.local_rank == 0:
            tfboard_writer.add_scalar("train/iter-lr", lr, all_iters)
            logger.info('It[{0}/{1}] Bt {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                        'Dt {data_time.val:.3f} ({data_time.avg:.3f}) Loss {loss.val:.3f} ({loss.avg:.3f}) '
                        'Prec@1 {top1.val:.3f} ({top1.avg:.3f}) Prec@5 {top5.val:.3f} ({top5.avg:.3f}) LR {lr:.3E}' \
                        .format(all_iters, args.total_iters, batch_time=batch_time, data_time=data_time,
                                loss=losses, top1=top1, top5=top5, lr=lr))
        
        if args.local_rank == 0:
            end = time.time()
        
    loss = torch.tensor([losses.avg]).cuda() if args.local_rank == 0 else torch.tensor([0.]).cuda()
    dist.broadcast(loss, 0)
    return loss.cpu().item(), all_iters

@torch.no_grad()
def validate(val_loader, model, all_iters=None):
    losses = AverageMeter()
    if args.local_rank == 0:
        batch_time = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()
    # switch to evaluate mode
    model.eval()
    if args.local_rank == 0:
        end = time.time()
    for i, data in enumerate(val_loader):
        target = data[0]["label"].squeeze().cuda().long()
        data = data[0]["data"]
        if args.fp16:
            data = data.half()
        # compute output
        output = model(data)
        loss = criterion_smooth(output, target)

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
                logger.info('Test Loss {loss.val:.3f} (avg={loss.avg:.3f}) Prec@1 {top1.val:.3f} '
                            '(avg={top1.avg:.3f}) Prec@5 {top5.val:.3f} (avg={top5.avg:.3f})' \
                            .format(loss=losses, top1=top1, top5=top5))

    if args.local_rank == 0:
        logger.info('Test: [{0}/{1}] Prec@1 {top1.avg:.5f} Prec@5 {top5.avg:.5f}' \
                    .format(all_iters, args.total_iters, top1=top1, top5=top5))

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

if __name__ == "__main__":
    main()


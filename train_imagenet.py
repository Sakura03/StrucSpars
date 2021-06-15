import torch, os, argparse, time, warnings
import numpy as np
from os.path import join, isfile
from vlutils import Logger, save_checkpoint, AverageMeter, accuracy, CosAnnealingLR
import resnet, densenet
from groupconv import GroupableConv2d
from utils import get_struc_reg_mat, get_perm_weight_norm, get_sparsity, get_threshold, synchronize_model, \
                  set_group_levels, update_permutation_matrix, mask_group, real_group, impose_group_lasso
import torch.nn.functional as F
from tensorboardX import SummaryWriter
from thop import profile
from thop.vision.basic_hooks import count_convNd
# DALI data reader
from nvidia.dali.pipeline import Pipeline
import nvidia.dali.ops as ops
import nvidia.dali.types as types
from nvidia.dali.plugin.pytorch import DALIClassificationIterator
# distributed
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
warnings.filterwarnings("ignore", "(Possibly )?corrupt EXIF data", UserWarning)
warnings.simplefilter('error')


parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N', help='manual epoch number (useful on restarts)')
parser.add_argument('--print-freq', default=50, type=int, metavar='N', help='print frequency (default: 50)')
parser.add_argument('-a', '--arch', type=str, metavar='STR', choices=["resnet50", "resnet101", "densenet201"], help='model architecture')
parser.add_argument('--data', metavar='DIR', default="./data", help='path to dataset')
parser.add_argument('--num-classes', default=1000, type=int, metavar='N', help='number of classes')
parser.add_argument('-j', '--workers', default=4, type=int, help='number of data loading workers')
parser.add_argument('--resume', default=None, type=str, metavar='PATH', help='path to the latest checkpoint')
parser.add_argument('--imsize', default=256, type=int, metavar='N', help='image resize size')
parser.add_argument('--imcrop', default=224, type=int, metavar='N', help='image crop size')
parser.add_argument('--save-path', default="results/tmp", type=str, help='path to save results')
# train
parser.add_argument('--batch-size', default=64, type=int, metavar='N', help='mini-batch size per GPU')
parser.add_argument('--epochs', default=60, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('--warmup', default=5, type=int, help="warmup epochs (small lr and do not impose sparsity)")
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float, metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='SGD momentum')
parser.add_argument('--weight-decay', '--wd', default=0.0, type=float, metavar='WD', help='weight decay')
# finetune
parser.add_argument('--ft-lr', default=0.1, type=int, help="initial learning rate in finetune stage")
parser.add_argument('--ft-epochs', default=120, type=int, help="number of total epochs to finetune")
parser.add_argument('--ft-warmup', default=5, type=int, help="warmup epochs in finetune stage")
parser.add_argument('--ft-wd', default=1e-4, type=float, help="weight decay in finetune stage")
# group
parser.add_argument('--delta-lambda', default=2e-6, type=float, help='L1 reg coefficient gain in each epoch')
parser.add_argument('--sparse-thres', default=0.1, type=float, help='sparse threshold (p)')
parser.add_argument('--decay-factor', default=0.5, type=float, help='decay factor in structured regularization matrix')
parser.add_argument('--prune-percent', default=0.5, type=float, help='parameter percent to prune')
# distributed
parser.add_argument("--local_rank", default=0, type=int)
parser.add_argument('--dali-cpu', action='store_true', help='run CPU-based version of DALI pipeline.')
args = parser.parse_args()


# DALI pipelines
class HybridTrainPipe(Pipeline):
    def __init__(self, batch_size, num_threads, device_id, data_dir, crop, dali_cpu=False):
        super(HybridTrainPipe, self).__init__(batch_size, num_threads, device_id, seed=12 + device_id)
        # MXnet rec reader
        self.input = ops.MXNetReader(path=join(data_dir, "train.rec"), index_path=join(data_dir, "train.idx"),
                                    random_shuffle=True, shard_id=args.local_rank, num_shards=args.world_size)
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
        self.input = ops.MXNetReader(path=join(data_dir, "val.rec"), index_path=join(data_dir, "val.idx"),
                                     random_shuffle=False, shard_id=args.local_rank, num_shards=args.world_size)
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
os.makedirs(args.save_path, exist_ok=True)
if args.local_rank == 0:
    tfboard_writer = SummaryWriter(log_dir=args.save_path)
    logger = Logger(join(args.save_path, "log.txt"))

# Customized operation for profiling
custom_ops = {GroupableConv2d: count_convNd}

# Set device
args.gpu = args.local_rank % torch.cuda.device_count()
torch.cuda.device(args.gpu)
local_device = torch.device("cuda", args.gpu)


def main():
    if args.local_rank == 0:
        logger.info(args)

    # Pytorch distributed setup
    dist.init_process_group(backend='nccl', init_method='env://')
    args.world_size = dist.get_world_size()

    # Build DALI dataloader
    pipe = HybridTrainPipe(batch_size=args.batch_size, num_threads=args.workers,
                           device_id=args.local_rank, data_dir=args.data, 
                           crop=args.imcrop, dali_cpu=args.dali_cpu)
    pipe.build()
    train_loader = DALIClassificationIterator(pipe, size=int(pipe.epoch_size("Reader") / args.world_size))

    pipe = HybridValPipe(batch_size=50, num_threads=args.workers,
                         device_id=args.local_rank, data_dir=args.data,
                         crop=args.imcrop, size=args.imsize)
    pipe.build()
    val_loader = DALIClassificationIterator(pipe, size=int(pipe.epoch_size("Reader") / args.world_size))
    train_loader_len = int(train_loader._size / args.batch_size)

    # Define model and optimizer
    prefix = "resnet." if "resnet" in args.arch else "densenet."
    model_name = prefix + "%s(num_classes=%d, power=%f)" % (args.arch, args.num_classes, args.decay_factor)
    model = eval(model_name).to(local_device)
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    if args.local_rank == 0:
        m = eval(model_name).to(local_device)
        logger.info("Model details:")
        logger.info(m)
        flops, params = profile(m, inputs=(torch.randn(1, 3, 224, 224).to(local_device),), custom_ops=custom_ops, verbose=False)
        del m
        tfboard_writer.add_scalar("train/FLOPs", flops, global_step=-1)
        tfboard_writer.add_scalar("train/Params", params, global_step=-1)
    model = DDP(model, device_ids=[args.local_rank], output_device=args.local_rank)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    if args.local_rank == 0:
        logger.info("Optimizer details:")
        logger.info(optimizer)

    # Save initial weights
    if args.local_rank == 0:
        save_checkpoint({
                'epoch': -1,
                'state_dict': model.state_dict(),
                'best_acc1': 0.0,
        }, is_best=False, path=args.save_path, filename="initial-weights.pth")

    # Optionally resume from a checkpoint
    if args.resume:
        assert isfile(args.resume), "=> no checkpoint found at '{}'".format(args.resume)
        if args.local_rank == 0:
            logger.info("=> loading checkpoint '{}'".format(args.resume))
        checkpoint = torch.load(args.resume, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint['state_dict'], strict=False)
        optimizer.load_state_dict(checkpoint['optimizer'])
        args.start_epoch = checkpoint['epoch']
        if args.local_rank == 0:
            best_acc1 = checkpoint['best_acc1']
            logger.info("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
    else:
        args.start_epoch = 0
        if args.local_rank == 0: best_acc1 = 0

    if args.local_rank == 0:
        # Log initial group info
        weight_norm = get_perm_weight_norm(model.module)
        struc_reg = get_struc_reg_mat(model.module)
        with torch.no_grad():
            for k in weight_norm.keys():
                wn = weight_norm[k] / (weight_norm[k].max() + 1e-8)
                sr = struc_reg[k]
                canvas = torch.cat((wn, torch.ones(wn.size(0), wn.size(1)//4).to(device=wn.device), sr.to(device=wn.device)), dim=1)
                tfboard_writer.add_image("train/%s" % k, canvas.unsqueeze(0), global_step=-2)
        # Initially update permutation matrices P and Q
        update_permutation_matrix(model, iters=50)

    # Synchronize grouping state (P/Q/P_inv/Q_inv/group_level/struc_reg_mat)
    synchronize_model(model)

    if args.local_rank == 0:
        # Log group info after initial permutation
        weight_norm = get_perm_weight_norm(model.module)
        struc_reg = get_struc_reg_mat(model.module)
        last_sparsity = get_sparsity(weight_norm, thres=args.sparse_thres)
        with torch.no_grad():
            for k in weight_norm.keys():
                wn = weight_norm[k] / (weight_norm[k].max() + 1e-8)
                sr = struc_reg[k]
                canvas = torch.cat((wn, torch.ones(wn.size(0), wn.size(1)//4).to(device=wn.device), sr), dim=1)
                tfboard_writer.add_image("train/%s" % k, canvas.unsqueeze(0), global_step=-1)

    struc_reg_coeff = args.delta_lambda
    for epoch in range(args.start_epoch, args.epochs):
        # Train and evaluate
        cur_reg_coeff = struc_reg_coeff if epoch >= args.warmup else 0.0
        loss = train(train_loader, model, optimizer, None, epoch, l1lambda=cur_reg_coeff)
        acc1, acc5 = validate(val_loader, model)

        if args.local_rank == 0:
            # Update permutation matrices P and Q
            update_permutation_matrix(model, iters=20)

            # Calculate current sparsity, FLOPs, and params
            weight_norm = get_perm_weight_norm(model.module)
            model_sparsity = get_sparsity(weight_norm, thres=args.sparse_thres)
            m = eval(model_name).to(local_device)
            group_levels = mask_group(m, weight_norm, args.sparse_thres, logger)
            real_group(m)
            flops, params = profile(m, inputs=(torch.randn(1, 3, 224, 224).to(local_device),), custom_ops=custom_ops, verbose=False)
            del m
            torch.cuda.empty_cache()
            logger.info("Sparsity %.6f, %.3e FLOPs, %.3e params" % (model_sparsity, flops, params))

            # Set group levels and gei current sparsity
            set_group_levels(model.module, group_levels)

            # Remember best prec@1 and save checkpoint
            is_best = acc1 > best_acc1
            if is_best: best_acc1 = acc1

            # Save checkpoint and log info
            save_checkpoint({
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'best_acc1': best_acc1,
                    'optimizer' : optimizer.state_dict()
            }, is_best, path=args.save_path)
            logger.info("Best acc1=%.5f" % best_acc1)

            struc_reg = get_struc_reg_mat(model.module)
            tfboard_writer.add_scalar("train/FLOPs", flops, epoch)
            tfboard_writer.add_scalar("train/Params", params, epoch)
            tfboard_writer.add_scalar('train/loss', loss, epoch)
            tfboard_writer.add_scalar('train/model-sparsity', model_sparsity, epoch)
            tfboard_writer.add_scalar('train/L1-coefficient', struc_reg_coeff, epoch)
            tfboard_writer.add_scalar('test/acc1', acc1, epoch)
            tfboard_writer.add_scalar('test/acc5', acc5, epoch)
            with torch.no_grad():
                for k in weight_norm.keys():
                    wn = weight_norm[k] / (weight_norm[k].max() + 1e-8)
                    sr = struc_reg[k]
                    canvas = torch.cat((wn, torch.ones(wn.size(0), wn.size(1)//4).to(device=wn.device), sr), dim=1)
                    tfboard_writer.add_image("train/%s" % k, canvas.unsqueeze(0), global_step=epoch)

            # Adjust coefficient of L1 regularization
            target_sparsity = args.prune_percent
            sparsity_gain = model_sparsity - last_sparsity
            expected_sparsity_gain = (target_sparsity - model_sparsity) / (args.epochs - epoch)
            if epoch >= args.warmup:
                # not sparse enough
                if model_sparsity < target_sparsity and sparsity_gain < expected_sparsity_gain:
                    logger.info("Sparsity gain %f (expected %f), increase reg coefficient by %.3e" % (sparsity_gain, expected_sparsity_gain, args.delta_lambda))
                    struc_reg_coeff += args.delta_lambda
                # over sparse
                elif model_sparsity >= target_sparsity:
                    struc_reg_coeff -= args.delta_lambda
                struc_reg_coeff = max(struc_reg_coeff, 0)
            logger.info("Model sparsity=%f (last=%f, target=%f), coefficient=%f" % (model_sparsity, last_sparsity, target_sparsity, cur_reg_coeff))
            last_sparsity = model_sparsity

        # Broadcast to other workers
        synchronize_model(model)           
        current_sparsity = torch.tensor([struc_reg_coeff]).to(local_device)
        dist.broadcast(current_sparsity, 0)
        struc_reg_coeff = current_sparsity.cpu().item()
        del current_sparsity

    # Evaluate before grouping
    if args.local_rank == 0:
        logger.info("Training done! Evaluating before grouping...")
    acc1, acc5 = validate(val_loader, model)
    if args.local_rank == 0:
        tfboard_writer.add_scalar('finetune/acc1', acc1, global_step=-2)
        tfboard_writer.add_scalar('finetune/acc5', acc5, global_step=-2)

    # Calculate group threshold, final sparsity, FLOPs, and params
    with torch.no_grad():
        thres = get_threshold(model.module, args.prune_percent)
    weight_norm = get_perm_weight_norm(model.module)
    if args.local_rank == 0:
        m = eval(model_name).to(local_device)
        group_levels = mask_group(m, weight_norm, thres, logger=None)
        real_group(m)
        set_group_levels(model.module, group_levels)

        model_sparsity = get_sparsity(weight_norm, thres=thres)
        flops, params = profile(m, inputs=(torch.randn(1, 3, 224, 224).to(local_device),), custom_ops=custom_ops, verbose=False)
        del m
        torch.cuda.empty_cache()
        logger.info("Threshold %.3e, final sparsity %.6f, target sparsity %.6f, %.3e FLOPs, %.3e params" % (thres, model_sparsity, args.prune_percent, flops, params))

    # Mask group
    group_levels = mask_group(model.module, weight_norm, thres, logger=logger if args.local_rank == 0 else None)
    if args.local_rank == 0:
        logger.info("Evaluating after mask grouping...")
    acc1, acc5 = validate(val_loader, model)

    # Real group
    # real_group(model.module)
    # if args.local_rank == 0:
    #     flops, params = profile(model.module, inputs=(torch.randn(1, 3, 32, 32).to(local_device),), custom_ops=custom_ops, verbose=False)
    #     logger.info("FLOPs %.3e, Params %.3e (after real grouping)" % (flops, params))
    #     logger.info("Evaluating after real grouping...")
    # acc1, acc5 = validate(val_loader, model)

    if args.local_rank == 0:
        tfboard_writer.add_scalar('finetune/acc1', acc1, global_step=-1)
        tfboard_writer.add_scalar('finetune/acc5', acc5, global_step=-1)

    # Start finetune stage
    optimizer = torch.optim.SGD(model.parameters(), lr=args.ft_lr, momentum=args.momentum, weight_decay=args.ft_wd)
    scheduler = CosAnnealingLR(loader_len=train_loader_len, epochs=args.ft_epochs, lr_max=args.ft_lr, warmup_epochs=args.ft_warmup)

    if args.local_rank == 0:
        best_acc1 = 0    
    for epoch in range(args.ft_epochs):
        # Train and evaluate
        loss = train(train_loader, model, optimizer, scheduler, epoch, finetune=True)
        acc1, acc5 = validate(val_loader, model)

        if args.local_rank == 0:
            # Remember best prec@1 and save checkpoint
            is_best = acc1 > best_acc1
            if is_best:
                best_acc1 = acc1

            # Save checkpoint and log info
            save_checkpoint({
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'best_acc1': best_acc1,
                    'optimizer' : optimizer.state_dict()
            }, is_best, path=args.save_path, filename="finetune-checkpoint.pth")
            logger.info("Best acc1=%.5f" % best_acc1)
            tfboard_writer.add_scalar('finetune/loss', loss, epoch)
            tfboard_writer.add_scalar('finetune/acc1', acc1, epoch)
            tfboard_writer.add_scalar('finetune/acc5', acc5, epoch)


def train(train_loader, model, optimizer, scheduler, epoch, l1lambda=0., finetune=False):
    if args.local_rank == 0:
        data_times, batch_times, losses, acc1, acc5 = [AverageMeter() for _ in range(5)]
        train_loader_len = int(np.ceil(train_loader._size/args.batch_size))

    # switch to train mode
    model.train()
    if args.local_rank == 0:
        end = time.time()
    for i, data in enumerate(train_loader):
        # Load data and distribute to devices
        image = data[0]["data"]
        target = data[0]["label"].squeeze().to(local_device).long()
        if args.local_rank == 0:
            start = time.time()
            # data_time.update(time.time() - end)

        # Compute the learning rate
        if scheduler is not None:
            lr = scheduler.step()
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

        # Forward training image and compute cross entropy loss
        prediction = model(image)
        loss = F.cross_entropy(prediction, target, reduction='mean')

        # One SGD step
        optimizer.zero_grad()
        loss.backward()
        if not finetune and l1lambda > 0.:
            impose_group_lasso(model, l1lambda)
        optimizer.step()

        # Compute accuracy
        top1, top5 = accuracy(prediction, target, topk=(1, 5))

        # Gather tensors from different devices
        loss = reduce_tensor(loss)
        top1 = reduce_tensor(top1)
        top5 = reduce_tensor(top5)

        # Update AverageMeter stats
        if args.local_rank == 0:
            data_times.update(start - end)
            batch_times.update(time.time() - start)
            losses.update(loss.item(), image.size(0))
            acc1.update(top1.item(), image.size(0))
            acc5.update(top5.item(), image.size(0))
        del prediction, loss, top1, top5
        torch.cuda.empty_cache()
        # torch.cuda.synchronize()

        # Update permutation matrices P and Q per 500 iters
        if not finetune and (i+1) % 500 == 0:
            if args.local_rank == 0:    
                update_permutation_matrix(model, iters=5)
            synchronize_model(model)

        # Log training info
        if i % args.print_freq == 0 and args.local_rank == 0:
            lr = optimizer.param_groups[0]["lr"]
            tfboard_writer.add_scalar("finetune/learning-rate" if finetune else "train/learning-rate", lr, epoch*train_loader_len+i)
            logger.info('Ep[{0}/{1}] It[{2}/{3}] Bt {batch_time.avg:.3f} Dt {data_time.avg:.3f} '
                        'Loss {loss.val:.3f} ({loss.avg:.3f}) Acc1 {top1.val:.3f} ({top1.avg:.3f}) '
                        'Acc5 {top5.val:.3f} ({top5.avg:.3f}) LR {lr:.3E} L1 {l1:.2E}'.format(
                                epoch, args.finetune_epochs if finetune else args.epochs,
                                i, train_loader_len, batch_time=batch_times, data_time=data_times,
                                loss=losses, top1=acc1, top5=acc5, lr=lr, l1=l1lambda
                        ))
        if args.local_rank == 0:
            end = time.time()

    # Reset training loader
    train_loader.reset()
    loss = torch.tensor([losses.avg]).to(local_device) if args.local_rank == 0 else torch.tensor([0.]).to(local_device)
    dist.broadcast(loss, 0)
    return loss.cpu().item()


@torch.no_grad()
def validate(val_loader, model):
    if args.local_rank == 0:
        losses, top1, top5 = [AverageMeter() for _ in range(3)]
        val_loader_len = int(np.ceil(val_loader._size/50))

    # Switch to evaluate mode
    model.eval()
    for i, data in enumerate(val_loader):
        image = data[0]["data"]
        target = data[0]["label"].squeeze().to(local_device).long()

        # Compute output
        prediction = model(image)
        loss = F.cross_entropy(prediction, target, reduction='mean')

        # Measure accuracy and record loss
        acc1, acc5 = accuracy(prediction, target, topk=(1, 5))

        # Gather tensors from different devices
        loss = reduce_tensor(loss)
        acc1 = reduce_tensor(acc1)
        acc5 = reduce_tensor(acc5)

        # Update meters and log info
        if args.local_rank == 0:
            losses.update(loss.item(), image.size(0))
            top1.update(acc1.item(), image.size(0))
            top5.update(acc5.item(), image.size(0))
        del prediction, loss, top1, top5
        torch.cuda.empty_cache()

        # Log validation info
        if i % args.print_freq == 0 and args.local_rank == 0:
            logger.info('Test: [{0}/{1}] Test Loss {loss.val:.3f} (avg={loss.avg:.3f}) '
                        'Acc1 {top1.val:.3f} (avg={top1.avg:.3f}) Acc5 {top5.val:.3f} (avg={top5.avg:.3f})' \
                        .format(i, val_loader_len, loss=losses, top1=top1, top5=top5))

    if args.local_rank == 0:
        logger.info(' * Prec@1 {top1.avg:.5f} Prec@5 {top5.avg:.5f}'.format(top1=top1, top5=top5))

    # Reset validation loader
    val_loader.reset()
    top1 = torch.tensor([top1.avg]).to(local_device) if args.local_rank == 0 else torch.tensor([0.]).to(local_device)
    top5 = torch.tensor([top5.avg]).to(local_device) if args.local_rank == 0 else torch.tensor([0.]).to(local_device)
    dist.broadcast(top1, 0)
    dist.broadcast(top5, 0)
    return top1.cpu().item(), top5.cpu().item()


def reduce_tensor(tensor):
    rt = tensor.clone()
    dist.reduce(rt, 0, op=dist.ReduceOp.SUM)
    rt /= args.world_size
    return rt


if __name__ == '__main__':
    main()

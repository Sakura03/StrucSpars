import torch, os, argparse, time
import numpy as np
from os.path import join, isfile, abspath
from vlutils import Logger, save_checkpoint, AverageMeter, accuracy, cifar10, cifar100, MultiStepLR
from model import * 
from utils import get_factors, get_sparsity, get_sparsity_loss, get_threshold, impose_group_lasso
from utils import set_group_levels, update_permutation_matrix, mask_group, real_group
from tensorboardX import SummaryWriter
from thop import profile, count_hooks

parser = argparse.ArgumentParser(description='PyTorch Cifar Training')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N', help='manual epoch number (useful on restarts)')
parser.add_argument('--print-freq', default=20, type=int, metavar='N', help='print frequency (default: 20)')
parser.add_argument('--depth', type=int, default=56, help='model depth')
parser.add_argument('--data', metavar='DIR', default="./data", help='path to dataset')
parser.add_argument('--dataset', default="cifar100", help='dataset')
parser.add_argument('--bs', '--batch-size', default=256, type=int, metavar='N', help='mini-batch size')
parser.add_argument('--epochs', default=160, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float, metavar='LR', help='initial learning rate')
parser.add_argument('--gamma', default=0.1, type=float, metavar='GM', help='decrease learning rate by gamma')
parser.add_argument('--milestones', default=[80, 120], type=eval, help='milestones for scheduling lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float, metavar='WD', help='weight decay')
parser.add_argument('--resume', default=None, type=str, metavar='PATH', help='path to the latest checkpoint')
parser.add_argument('--tmp', default="results/tmp", type=str, help='tmp folder')
parser.add_argument('--randseed', type=int, default=None, help='random seed')
parser.add_argument('--fix-lr', action="store_true", help='set true to fix learning rate')
parser.add_argument('--no-finetune', action="store_true", help='set true to disable finetuning')
parser.add_argument('--adjust-lambda', action="store_true", help='set true to automatically adjust l1lambda')
parser.add_argument('--sparsity', type=float, default=2e-6, help='sparsity regularization')
parser.add_argument('--delta-lambda', type=float, default=2e-6, help='delta lambda')
parser.add_argument('--sparse-thres', type=float, default=0.1, help='sparse threshold')
parser.add_argument('--finetune-lr', type=float, default=0.1, help="finetune learning rate")
parser.add_argument('--finetune-epochs', type=int, default=160, help="finetune epochs")
parser.add_argument('--finetune-milestones', default=[80, 120], type=eval, help='milestones for scheduling lr in finetuning')
parser.add_argument('--finetune-weight-decay', type=float, default=1e-4, help="finetune weight decay")
parser.add_argument('--init-iters', type=int, default=20, help='Initial iterations')
parser.add_argument('--epoch-iters', type=int, default=10, help='Iterations for each epoch')
parser.add_argument('--warmup', type=int, default=10, help='Warmup epochs (do not adjust lambda)')
parser.add_argument('--power', type=float, default=0.5, help='Decay rate in the penalty matrix')
parser.add_argument('--percent', type=float, default=0.5, help='remaining parameter percent')
args = parser.parse_args()

if args.randseed == None:
    args.randseed = np.random.randint(1000)
args.tmp = args.tmp.strip("/")
args.tmp = join(args.tmp, "seed%d" % args.randseed)

# Random seed
# According to https://pytorch.org/docs/master/notes/randomness.html
np.random.seed(args.randseed)
torch.manual_seed(args.randseed)
torch.cuda.manual_seed_all(args.randseed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

os.makedirs(args.tmp, exist_ok=True)
os.makedirs(join(args.tmp, "factors"), exist_ok=True)

# loss function
criterion = torch.nn.CrossEntropyLoss()
custom_ops = {GroupableConv2d: count_hooks.count_convNd}

tfboard_writer = SummaryWriter(log_dir=args.tmp)
logger = Logger(join(args.tmp, "log.txt"))

def main():
    logger.info(args)
    if args.dataset == "cifar10":
        train_loader, val_loader = cifar10(abspath(args.data), bs=args.bs)
        args.num_classes = 10
    elif args.dataset == "cifar100":
        train_loader, val_loader = cifar100(abspath(args.data), bs=args.bs)
        args.num_classes = 100

    # model and optimizer
    model_name = "resnet%d_cifar(num_classes=%d, groupable=True)" % (args.depth, args.num_classes)
    model = eval(model_name).cuda()
    flops, params = profile(model, inputs=(torch.randn(1, 3, 32, 32).cuda(), ), custom_ops=custom_ops, verbose=False)
    tfboard_writer.add_scalar("train/FLOPs", flops, global_step=-1)
    tfboard_writer.add_scalar("train/Params", params, global_step=-1)
    # model = nn.DataParallel(model)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    logger.info("Model details:")
    logger.info(model)
    logger.info("Optimizer details:")
    logger.info(optimizer)

    # records
    best_acc1 = 0

    # save initial weights
    save_checkpoint({
            'epoch': -1,
            'state_dict': model.state_dict(),
            'best_acc1': -1,
            }, is_best=False, path=args.tmp, filename="initial-weights.pth")

    # optionally resume from a checkpoint
    if args.resume is not None:
        if isfile(args.resume):
            logger.info("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            logger.info("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
        else:
            logger.info("=> no checkpoint found at '{}'".format(args.resume))

    scheduler = None if args.fix_lr else MultiStepLR(len(train_loader), milestones=args.milestones, gamma=args.gamma)
    
    # before initial update
    factors = get_factors(model)
    torch.save(factors, join(args.tmp, "factors", "before-permutation.pth"))
    for k, v in factors.items():
        tfboard_writer.add_image("train/%s" % k, v.unsqueeze(0) / (v.max()+1e-8), global_step=-1)
    # initially update P and Q
    update_permutation_matrix(model, iters=args.init_iters)
    # after initial update
    factors = get_factors(model)
    torch.save(factors, join(args.tmp, "factors", "after-permutation.pth"))
    for k, v in factors.items():
        tfboard_writer.add_image("train/%s" % k, v.unsqueeze(0) / (v.max()+1e-8), global_step=-1)
    
    if args.adjust_lambda:
        last_sparsity = get_sparsity(factors, thres=args.sparse_thres)
    for epoch in range(args.start_epoch, args.epochs):
        # train and evaluate
        loss = train(train_loader, model, optimizer, scheduler, epoch, l1lambda=args.sparsity if epoch >= args.warmup else 0.)
        acc1, acc5 = validate(val_loader, model)
        
        # update P and Q
        update_permutation_matrix(model, iters=args.epoch_iters)
        
        # compute the regularity
        sloss = get_sparsity_loss(model, enable_grad=False)
        
        # calculate sparsity, FLOPs and params
        m = eval(model_name).cuda()
        factors = get_factors(model)
        group_levels = mask_group(m, factors, args.sparse_thres, logger)
        real_group(m)
        set_group_levels(model, group_levels)
        
        model_sparsity = get_sparsity(factors, thres=args.sparse_thres)
        flops, params = profile(m, inputs=(torch.randn(1, 3, 32, 32).cuda(), ), custom_ops=custom_ops, verbose=False)
        del m
        torch.cuda.empty_cache()
        logger.info("Sparsity %.6f, %.3e FLOPs, %.3e params" % (model_sparsity, flops, params))

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
        
        # optionally adjust l1lambda
        if args.adjust_lambda:
            target_sparsity = args.percent
            sparsity_gain = (model_sparsity - last_sparsity)
            expected_sparsity_gain = (target_sparsity - model_sparsity) / (args.epochs - epoch)
            if epoch >= args.warmup:
                # not sparse enough
                if model_sparsity < target_sparsity and sparsity_gain < expected_sparsity_gain:
                    logger.info("Sparsity gain %f (expected %f), increasing the sparsity penalty." % (sparsity_gain, expected_sparsity_gain))
                    args.sparsity += args.delta_lambda
                # over sparse
                elif model_sparsity >= target_sparsity:
                    args.sparsity -= args.delta_lambda
                # minimal sparsity=0
                args.sparsity = max(args.sparsity, 0)
            logger.info("Model sparsity=%f (last=%f, target=%f), args.sparsity=%f" % (model_sparsity, last_sparsity, target_sparsity, args.sparsity))
            last_sparsity = model_sparsity
                
        tfboard_writer.add_scalar("train/FLOPs", flops, epoch)
        tfboard_writer.add_scalar("train/Params", params, epoch)
        tfboard_writer.add_scalar('train/loss-epoch', loss, epoch)
        tfboard_writer.add_scalar('train/sloss-epoch', sloss, epoch)
        tfboard_writer.add_scalar('train/model-sparsity', model_sparsity, epoch)
        tfboard_writer.add_scalar('train/sparse-penalty', args.sparsity, epoch)
        tfboard_writer.add_scalar('test/acc1-epoch', acc1, epoch)
        tfboard_writer.add_scalar('test/acc5-epoch', acc5, epoch)

        torch.save(factors, join(args.tmp, "factors", "epoch%03d.pth"%epoch))
        for k, v in factors.items():
            tfboard_writer.add_image("train/%s" % k, v.unsqueeze(0) / (v.max()+1e-8), global_step=epoch)

    logger.info("Training done, ALL results saved to %s." % args.tmp)

    # evaluate before grouping
    logger.info("evaluating before grouping...")
    acc1, acc5 = validate(val_loader, model)
    tfboard_writer.add_scalar('finetune/acc1-epoch', acc1, global_step=-2)
    tfboard_writer.add_scalar('finetune/acc5-epoch', acc5, global_step=-2)

    # mask grouping
    thres = args.sparse_thres
    with torch.no_grad():    
        thres = get_threshold(model, args.percent)
    # calculate final sparsity, FLOPs, and params
    factors = get_factors(model)
    m = eval(model_name).cuda()
    group_levels = mask_group(m, factors, thres, logger=None)
    real_group(m)
    set_group_levels(model, group_levels)
   
    model_sparsity = get_sparsity(factors, thres=thres)
    flops, params = profile(m, inputs=(torch.randn(1, 3, 32, 32).cuda(),), custom_ops=custom_ops, verbose=False)
    del m
    torch.cuda.empty_cache()
    logger.info("Threshold %.3e, final sparsity %.6f, target sparsity %.6f, %.3e FLOPs, %.3e params" % (thres, model_sparsity, args.percent, flops, params))

    group_levels = mask_group(model, factors, thres, logger=logger)
    torch.save(group_levels, join(args.tmp, "group_levels.pth"))
    
    logger.info("evaluating after grouping...")
    validate(val_loader, model)

    # real grouping
    # real_group(model)
    # flops, params = profile(model, inputs=(torch.randn(1, 3, 32, 32).cuda(), ), custom_ops=custom_ops, verbose=False)
    # logger.info("FLOPs %.3e, Params %.3e (after real grouping)" % (flops, params))

    # logger.info("evaluating after real grouping...")
    # acc1, acc5 = validate(val_loader, model, args.epochs)

    # shutdown when "args.no-finetune" is triggered
    if args.no_finetune: return

    tfboard_writer.add_scalar('finetune/acc1_epoch', acc1, global_step=-1)
    tfboard_writer.add_scalar('finetune/acc5_epoch', acc5, global_step=-1)

    # finetune
    optimizer_finetune = torch.optim.SGD(model.parameters(), lr=args.finetune_lr, momentum=args.momentum, weight_decay=args.finetune_weight_decay)

    scheduler_finetune = MultiStepLR(len(train_loader), milestones=args.finetune_milestones, gamma=args.gamma)

    best_acc1 = 0
    for epoch in range(0, args.finetune_epochs):
        # train and evaluate
        loss = train(train_loader, model, optimizer_finetune, scheduler_finetune, epoch, finetune=True)
        acc1, acc5 = validate(val_loader, model)

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
                }, is_best, path=args.tmp, filename="checkpoint-finetune.pth")

def train(train_loader, model, optimizer, scheduler, epoch, l1lambda=0., finetune=False):
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
        data = data.cuda()
        target = target.cuda()
        data_time.update(time.time() - end)

        output = model(data)
        loss = criterion(output, target)
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), data.size(0))
        top1.update(acc1.item(), data.size(0))
        top5.update(acc5.item(), data.size(0))
        
        # compute and adjust lr
        if scheduler is not None:
            lr = scheduler.step()
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        if not finetune and l1lambda > 0.:
            impose_group_lasso(model, l1lambda)
        optimizer.step()
        
        # torch.cuda.synchronize()
        # measure elapsed time
        batch_time.update(time.time() - end)
        lr = optimizer.param_groups[0]["lr"]
        
        if i % args.print_freq == 0:
            logger.info('Ep[{0}/{1}] It[{2}/{3}] Bt {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                        'Dt {data_time.val:.3f} ({data_time.avg:.3f}) Loss {loss.val:.3f} ({loss.avg:.3f}) '
                        'Prec@1 {top1.val:.3f} ({top1.avg:.3f}) Prec@5 {top5.val:.3f} ({top5.avg:.3f}) LR {lr:.3E} L1 {l1:.2E}' \
                        .format(epoch, args.finetune_epochs, i, len(train_loader), batch_time=batch_time, data_time=data_time,
                                loss=losses, top1=top1, top5=top5, lr=lr, l1=l1lambda))
        end = time.time()
    return losses.avg

@torch.no_grad()
def validate(val_loader, model):
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

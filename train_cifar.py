import torch, os, argparse, time
import numpy as np
from os.path import join, isfile, abspath
from vlutils import Logger, save_checkpoint, AverageMeter, accuracy, cifar10, cifar100, MultiStepLR
import resnet_cifar
from groupconv import GroupableConv2d
from utils import get_struc_reg_mat, get_perm_weight_norm, get_sparsity, get_threshold, \
                  set_group_levels, update_permutation_matrix, mask_group, real_group, impose_group_lasso
import torch.nn.functional as F
from tensorboardX import SummaryWriter
from thop import profile
from thop.vision.basic_hooks import count_convNd

parser = argparse.ArgumentParser(description='PyTorch CIFAR Training')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N', help='manual epoch number (useful on restarts)')
parser.add_argument('--print-freq', default=100, type=int, metavar='N', help='print frequency (default: 20)')
parser.add_argument('-a', '--arch', default="resnet20", type=str, metavar='STR', choices=["resnet20", "resnet56", "resnet110"], help='model architecture')
parser.add_argument('--data', default="./data", metavar='DIR', help='path to dataset')
parser.add_argument('--dataset', default="cifar10", type=str, choices=["cifar10", "cifar100"], help='dataset')
parser.add_argument('--resume', default=None, type=str, metavar='PATH', help='path to the latest checkpoint')
parser.add_argument('--save-path', default="results/tmp", type=str, help='path to save results')
parser.add_argument('--seed', default=None, type=int, help='random seed')
# train
parser.add_argument('--batch-size', default=64, type=int, metavar='N', help='mini-batch size')
parser.add_argument('--epochs', default=100, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('--warmup', default=10, type=int, help="warmup epochs (small lr and do not impose sparsity)")
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float, metavar='LR', help='initial learning rate')

parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='SGD momentum')
parser.add_argument('--weight-decay', '--wd', default=0.0, type=float, metavar='WD', help='weight decay')
# finetune
parser.add_argument('--ft-lr', default=0.1, type=int, help="initial learning rate in finetune stage")
parser.add_argument('--ft-epochs', default=160, type=int, help="number of total epochs to finetune")
parser.add_argument('--ft-warmup', default=10, type=int, help="warmup epochs in finetune stage")
parser.add_argument('--ft-milestones', default=[80, 120], type=eval, help='milestones for multi-step scheduler in finetune stage')
parser.add_argument('--ft-gamma', default=0.1, type=float, metavar='GM', help='decrease lr by gamma in finetune stage')
parser.add_argument('--ft-wd', default=1e-4, type=float, help="weight decay in finetune stage")
# group
parser.add_argument('--delta-lambda', default=2e-6, type=float, help='L1 reg coefficient gain in each epoch')
parser.add_argument('--sparse-thres', default=0.1, type=float, help='sparse threshold (p)')
parser.add_argument('--decay-factor', default=0.5, type=float, help='decay factor in structured regularization matrix')
parser.add_argument('--prune-percent', default=0.5, type=float, help='parameter percent to prune')
args = parser.parse_args()

# Set random seed
if args.seed == None:
    args.seed = np.random.randint(1000)
# According to https://pytorch.org/docs/master/notes/randomness.html
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Save dir
args.save_path = join(args.save_path, "seed%d" % args.seed)
os.makedirs(args.save_path, exist_ok=True)

# Customized operation for profiling
custom_ops = {GroupableConv2d: count_convNd}

tfboard_writer = SummaryWriter(log_dir=args.save_path)
logger = Logger(join(args.save_path, "log.txt"))

def main():
    logger.info(args)
    # Construct dataset
    if args.dataset == "cifar10":
        train_loader, val_loader = cifar10(abspath(args.data), bs=args.batch_size)
        args.num_classes = 10
    elif args.dataset == "cifar100":
        train_loader, val_loader = cifar100(abspath(args.data), bs=args.batch_size)
        args.num_classes = 100

    # Define model and optimizer
    model_name = "resnet_cifar.%s(num_classes=%d, power=%f)" % (args.arch, args.num_classes, args.decay_factor)
    model = eval(model_name).cuda()

    # Log initial complexity
    m = eval(model_name).cuda()
    logger.info("Model details:")
    logger.info(m)
    flops, params = profile(m, inputs=(torch.randn(1, 3, 32, 32).cuda(),), custom_ops=custom_ops, verbose=False)
    del m
    torch.cuda.empty_cache()
    tfboard_writer.add_scalar("train/FLOPs", flops, global_step=-1)
    tfboard_writer.add_scalar("train/Params", params, global_step=-1)

    # Optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    logger.info("Optimizer details:")
    logger.info(optimizer)

    # Save initial weights
    save_checkpoint({
            'epoch': -1,
            'state_dict': model.state_dict(),
            'best_acc1': 0.0,
    }, is_best=False, path=args.save_path, filename="initial-weights.pth")

    # Optionally resume from a checkpoint
    if args.resume:
        assert isfile(args.resume), "=> no checkpoint found at '{}'".format(args.resume)
        logger.info("=> loading checkpoint '{}'".format(args.resume))
        checkpoint = torch.load(args.resume, map_location=torch.device('cpu'))
        args.start_epoch = checkpoint['epoch']
        best_acc1 = checkpoint['best_acc1']
        model.load_state_dict(checkpoint['state_dict'], strict=False)
        optimizer.load_state_dict(checkpoint['optimizer'])
        logger.info("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
    else:
        args.start_epoch = 0
        best_acc1 = 0

    # Log initial group info
    weight_norm = get_perm_weight_norm(model)
    struc_reg = get_struc_reg_mat(model)
    with torch.no_grad():
        for k in weight_norm.keys():
            wn = weight_norm[k] / (weight_norm[k].max() + 1e-8)
            sr = struc_reg[k]
            canvas = torch.cat((wn, torch.ones(wn.size(0), wn.size(1)//4).to(device=wn.device), sr.to(device=wn.device)), dim=1)
            tfboard_writer.add_image("train/%s" % k, canvas.unsqueeze(0), global_step=-2)
    # Initially update permutation matrices P and Q
    update_permutation_matrix(model, iters=20)

    # Log group info after initial permutation
    weight_norm = get_perm_weight_norm(model)
    struc_reg = get_struc_reg_mat(model)
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

        # Update permutation matrices P and Q
        update_permutation_matrix(model, iters=10)

        # Calculate current sparsity, FLOPs, and params
        weight_norm = get_perm_weight_norm(model)
        model_sparsity = get_sparsity(weight_norm, thres=args.sparse_thres)
        m = eval(model_name).cuda()
        group_levels = mask_group(m, weight_norm, args.sparse_thres, logger)
        real_group(m)
        flops, params = profile(m, inputs=(torch.randn(1, 3, 32, 32).cuda(),), custom_ops=custom_ops, verbose=False)
        del m
        torch.cuda.empty_cache()
        logger.info("Sparsity %.6f, %.3e FLOPs, %.3e params" % (model_sparsity, flops, params))

        # Set group levels and gei current sparsity
        set_group_levels(model, group_levels)

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

        struc_reg = get_struc_reg_mat(model)
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

    # Evaluate before grouping
    logger.info("Training done! Evaluating before grouping...")
    acc1, acc5 = validate(val_loader, model)
    tfboard_writer.add_scalar('finetune/acc1', acc1, global_step=-2)
    tfboard_writer.add_scalar('finetune/acc5', acc5, global_step=-2)

    # Calculate group threshold, final sparsity, FLOPs, and params
    with torch.no_grad():
        thres = get_threshold(model, args.prune_percent)
    weight_norm = get_perm_weight_norm(model)
    m = eval(model_name).cuda()
    group_levels = mask_group(m, weight_norm, thres, logger=None)
    real_group(m)
    set_group_levels(model, group_levels)

    model_sparsity = get_sparsity(weight_norm, thres=thres)
    flops, params = profile(m, inputs=(torch.randn(1, 3, 224, 224).cuda(),), custom_ops=custom_ops, verbose=False)
    del m
    torch.cuda.empty_cache()
    logger.info("Threshold %.3e, final sparsity %.6f, target sparsity %.6f, %.3e FLOPs, %.3e params" % (thres, model_sparsity, args.prune_percent, flops, params))

    # Mask group
    group_levels = mask_group(model, weight_norm, thres, logger=logger)
    logger.info("Evaluating after mask grouping...")
    acc1, acc5 = validate(val_loader, model)

    # Real group
    # real_group(model)
    # flops, params = profile(model, inputs=(torch.randn(1, 3, 32, 32).cuda(),), custom_ops=custom_ops, verbose=False)
    # logger.info("FLOPs %.3e, Params %.3e (after real grouping)" % (flops, params))
    # logger.info("Evaluating after real grouping...")
    # acc1, acc5 = validate(val_loader, model)

    tfboard_writer.add_scalar('finetune/acc1', acc1, global_step=-1)
    tfboard_writer.add_scalar('finetune/acc5', acc5, global_step=-1)

    # Start finetune stage
    optimizer = torch.optim.SGD(model.parameters(), lr=args.ft_lr, momentum=args.momentum, weight_decay=args.ft_wd)
    scheduler = MultiStepLR(loader_len=len(train_loader), milestones=args.ft_milestones, gamma=args.ft_gamma, base_lr=args.ft_lr, warmup_epochs=args.ft_warmup)

    best_acc1 = 0
    for epoch in range(args.ft_epochs):
        # Train and evaluate
        loss = train(train_loader, model, optimizer, scheduler, epoch, finetune=True)
        acc1, acc5 = validate(val_loader, model)

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
    data_times, batch_times, losses, acc1, acc5 = [AverageMeter() for _ in range(5)]
    train_loader_len = len(train_loader)

    # switch to train mode
    model.train()
    end = time.time()
    for i, (image, target) in enumerate(train_loader):
        # Load data and distribute to devices
        image = image.cuda()
        target = target.cuda()
        start = time.time()

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

        # Update AverageMeter stats
        data_times.update(start - end)
        batch_times.update(time.time() - start)
        losses.update(loss.item(), image.size(0))
        acc1.update(top1.item(), image.size(0))
        acc5.update(top5.item(), image.size(0))

        # Log training info
        if i % args.print_freq == 0:
            lr = optimizer.param_groups[0]["lr"]
            tfboard_writer.add_scalar("finetune/learning-rate" if finetune else "train/learning-rate", lr, epoch*train_loader_len+i)
            logger.info('Ep[{0}/{1}] It[{2}/{3}] Bt {batch_time.avg:.3f} Dt {data_time.avg:.3f} '
                        'Loss {loss.val:.3f} ({loss.avg:.3f}) Acc1 {top1.val:.3f} ({top1.avg:.3f}) '
                        'Acc5 {top5.val:.3f} ({top5.avg:.3f}) LR {lr:.3E} L1 {l1:.2E}'.format(
                                epoch, args.ft_epochs if finetune else args.epochs,
                                i, train_loader_len, batch_time=batch_times, data_time=data_times,
                                loss=losses, top1=acc1, top5=acc5, lr=lr, l1=l1lambda
                        ))
        end = time.time()

    return losses.avg


@torch.no_grad()
def validate(val_loader, model):
    losses, top1, top5 = [AverageMeter() for _ in range(3)]
    val_loader_len = len(val_loader)

    # Switch to evaluate mode
    model.eval()
    for i, (image, target) in enumerate(val_loader):
        image = image.cuda()
        target = target.cuda()

        # Compute output
        prediction = model(image)
        loss = F.cross_entropy(prediction, target, reduction='mean')

        # Measure accuracy and record loss
        acc1, acc5 = accuracy(prediction, target, topk=(1, 5))

        # Update meters and log info
        losses.update(loss.item(), image.size(0))
        top1.update(acc1.item(), image.size(0))
        top5.update(acc5.item(), image.size(0))

        # Log validation info
        if i % args.print_freq == 0:
            logger.info('Test: [{0}/{1}] Test Loss {loss.val:.3f} (avg={loss.avg:.3f}) '
                        'Acc1 {top1.val:.3f} (avg={top1.avg:.3f}) Acc5 {top5.val:.3f} (avg={top5.avg:.3f})' \
                        .format(i, val_loader_len, loss=losses, top1=top1, top5=top5))

    logger.info(' * Prec@1 {top1.avg:.5f} Prec@5 {top5.avg:.5f}'.format(top1=top1, top5=top5))

    return top1.avg, top5.avg


if __name__ == '__main__':
    main()

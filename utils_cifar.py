import time, torch, multiprocessing
import numpy as np
from resnet_cifar import GroupableConv2d, get_penalty_matrix

@torch.no_grad()
def get_level(matrix, thres):
    matrix = matrix.clone()
    penalty = get_penalty_matrix(matrix.size(0), matrix.size(1))
    u = torch.unique(penalty, sorted=True)
    sums = [torch.sum(matrix)]
    for level in range(1, u.size(0)):
        mask = (penalty == u[-level])
        matrix[mask] = 0.
        sums.append(torch.sum(matrix))
        if sums[-1] / sums[-2] <= 1. / (1. + thres):
            break
    return level

@torch.no_grad()
def get_factors(model):
    factors = {}
    for name, m in model.named_modules():
        if isinstance(m, GroupableConv2d):
            weight_norm = torch.norm(m.weight.data.view(m.out_channels, m.in_channels, -1), dim=-1, p=1)
            factors[name] = weight_norm[m.P, :][:, m.Q]
    return factors

@torch.no_grad()
def get_sparsity(factors, thres):
    total0 = 0
    total = 0
    for v in factors.values():
        total0 += v.numel() / (2 ** (get_level(v, thres)-1))
        total += v.numel()
    return 1. - float(total0) / total

@torch.no_grad()
def get_sparsity_from_model(model, thres):
    return get_sparsity(get_factors(model), thres)

def get_sparsity_loss(model):
    loss = 0
    for m in model.modules():
        if isinstance(m, GroupableConv2d):
            loss += m.compute_regularity()
    return loss
            
@torch.no_grad()
def get_threshold(model, target_sparsity, head=0., tail=1., margin=0.001):
    sparsity = get_sparsity_from_model(model, thres=(head+tail)/2)
    if abs(sparsity - target_sparsity) <= margin:
        # the ONLY output port
        return (head+tail)/2
    else:
        if sparsity >= target_sparsity:
            return get_threshold(model, target_sparsity, head=head, tail=(head+tail)/2)
        else:
            return get_threshold(model, target_sparsity, head=(head+tail)/2, tail=tail)

def set_group_levels(model, group_levels):
    for name, m in model.named_modules():
        if isinstance(m, GroupableConv2d):
            m.set_group_level(group_levels[name])

def update_one_module(module, iters, name):
    P, Q = module.update_PQ(iters)
    # print("Update permutation matrix of %s" % name)
    return P, Q

def update_permutation_matrix(model, iters=1):
    start = time.time()
    model.cpu()
    pool = multiprocessing.Pool(processes=8)
    results = {}
    for name, m in model.named_modules():
        if isinstance(m, GroupableConv2d):
            results[name] = pool.apply_async(update_one_module, args=(m, iters, name, ))
    pool.close()
    pool.join()
        
    for name, m in model.named_modules():
        if isinstance(m, GroupableConv2d):
            m.P, m.Q = results[name].get()
    model.cuda()
    print("Update permutation matrices for %d iters, elapsed time: %.3f" % (iters, time.time()-start))

@torch.no_grad()
def mask_group(model, factors, thres, logger):
    group_levels = {}
    total_connections = 0
    remaining_connections = 0
    for name, m in model.named_modules():
        if isinstance(m, GroupableConv2d):
            level = get_level(factors[name], thres)
            group_levels[name] = level
            
            m.set_group_level(level)
            mask = m.mask_group()
            total_connections += mask.numel()
            remaining_connections += mask.sum()
            logger.info("Layer %s total connections %d (remaining %d)" % (name, mask.numel(), mask.sum()))
    logger.info("--------------------> %d of %d connections remained, remaining rate %f <--------------------" % \
               (remaining_connections, total_connections, float(remaining_connections)/total_connections))
    return group_levels

@torch.no_grad()
def real_group(model):
    for name, m in model.named_modules():
        if isinstance(m, GroupableConv2d):
            m.real_group()

if __name__ == "__main__":
    from resnet_cifar import resnet50
    import time
    model = resnet50(group1x1=True)
    start = time.time()
    update_permutation_matrix(model, iters=5000, mp=True)
    print(time.time() - start)

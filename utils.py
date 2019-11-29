import torch
import torch.nn as nn
from resnet_cifar import GroupableConv2d

def isPower(n):
    if n < 1:
        return False
    i = 1
    while i <= n:
        if i == n:
            return True
        i <<= 1
    return False

@torch.no_grad()
def get_penalty_matrix(dim1, dim2, power=0.5):
    assert isPower(dim1) and isPower(dim2)
    weight = torch.zeros(dim1, dim2).cuda()
    assign_location(weight, 1., power)
    return weight

@torch.no_grad()
def assign_location(tensor, num, power):
    dim1, dim2 = tensor.size()
    if dim1 == 1 or dim2 == 1:
        return
    else:
        tensor[dim1//2:, :dim2//2] = num
        tensor[:dim1//2, dim2//2:] = num
        assign_location(tensor[dim1//2:, dim2//2:], num*power, power)
        assign_location(tensor[:dim1//2, :dim2//2], num*power, power)

@torch.no_grad()
def get_level(matrix, thres):
    matrix = matrix.clone()
    penalty = get_penalty_matrix(matrix.size(0), matrix.size(1))
    u = torch.unique(penalty, sorted=True)
    sums = [torch.sum(torch.abs(matrix))]
    for level in range(1, u.size(0)):
        mask = (penalty == u[-level])
        matrix[mask] = 0.
        sums.append(torch.sum(torch.abs(matrix)))
        if sums[-1] / sums[-2] <= 1. / (1. + thres):
            break
    return level

@torch.no_grad()
def get_factors(model):
    factors = {}
    for name, m in model.named_modules():
        if isinstance(m, GroupableConv2d):
            m.compute_weight_norm()
            factors[name] = m.weight_norm[m.P, :][:, m.Q]
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

def get_sparsity_loss(model, penalties):
    loss = 0
    for m in model.modules():
        if isinstance(m, GroupableConv2d):
            dim = m.in_channels
            loss += m.compute_regularity(penalties[dim])
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

@torch.no_grad()
def get_mask_from_level(level, dim):
    mask = torch.ones(dim, dim).cuda()
    penalty = get_penalty_matrix(dim, dim)
    u = torch.unique(penalty, sorted=True)
    for i in range(1, level):
        mask[penalty == u[-i]] = 0.
    return mask

def update_permutation_matrix(model, penalties, iters=1):
    for name, m in model.named_modules():
        if isinstance(m, GroupableConv2d):
            if iters > 1:
                print("Iterate over layer %s" % name)
            dim = m.in_channels
            m.compute_weight_norm()
            m.stochastic_exchange(penalties[dim], iters)
       
@torch.no_grad()
def mask_group(model, factors, thres, logger):
    group_levels = {}
    total_connections = 0
    remaining_connections = 0
    for name, m in model.named_modules():
        if isinstance(m, GroupableConv2d):
            level = get_level(factors[name], thres)
            mask = get_mask_from_level(level, dim=m.in_channels)
            _, P_inv = torch.sort(m.P)
            _, Q_inv = torch.sort(m.Q)
            permuted_mask = mask[P_inv, :][:, Q_inv]
            permuted_mask.unsqueeze_(dim=-1).unsqueeze_(dim=-1)
            m.weight.data *= permuted_mask
            group_levels[name] = level
            total_connections += mask.numel()
            remaining_connections += mask.sum()
            logger.info("Layer %s total connections %d (remaining %d)" % (name, mask.numel(), mask.sum()))
    logger.info("--------------------> %d of %d connections remained, remaining rate %f <--------------------" % \
               (remaining_connections, total_connections, float(remaining_connections)/total_connections))
    return group_levels

@torch.no_grad()
def real_group(model, group_levels):
    for name, m in model.named_modules():
        if isinstance(m, GroupableConv2d):
            m.real_group(group_levels[name])

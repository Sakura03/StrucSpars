import torch
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

@torch.no_grad()
def get_mask_from_level(level, dim1, dim2):
    mask = torch.ones(dim1, dim2).cuda()
    penalty = get_penalty_matrix(dim1, dim2)
    u = torch.unique(penalty, sorted=True)
    for i in range(1, level):
        mask[penalty == u[-i]] = 0.
    return mask

def update_permutation_matrix(model, iters=1):
    for name, m in model.named_modules():
        if isinstance(m, GroupableConv2d):
            if iters > 1:
                print("Iterate over layer %s" % name)
            m.compute_weight_norm()
            m.stochastic_exchange(iters)
       
@torch.no_grad()
def mask_group(model, factors, thres, logger):
    group_levels = {}
    total_connections = 0
    remaining_connections = 0
    for name, m in model.named_modules():
        if isinstance(m, GroupableConv2d):
            level = get_level(factors[name], thres)
            mask = get_mask_from_level(level, m.in_channels, m.out_channels)
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

import torch, multiprocessing
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

@torch.no_grad()
def update_one_module_(module, iters):
    weight_norm = torch.norm(module.weight.data.view(module.out_channels, module.in_channels, -1), dim=-1, p=1)
    for i in range(iters):
        idx1, idx2 = np.random.choice(module.out_channels, size=2, replace=False)
        if compare_loss_(module.P, module.Q, idx1, idx2, weight_norm, module.penalty, row=True):
            tmp = module.P[idx1].clone()
            module.P[idx1] = module.P[idx2]
            module.P[idx2] = tmp
        idx1, idx2 = np.random.choice(module.in_channels, size=2, replace=False)
        if compare_loss_(module.P, module.Q, idx1, idx2, weight_norm, module.penalty, row=False):
            tmp = module.Q[idx1].clone()
            module.Q[idx1] = module.Q[idx2]
            module.Q[idx2] = tmp

@torch.no_grad()
def compare_loss_(P, Q, idx1, idx2, weight_norm, penalty, row=True):
    if row:
        shuffled_weight_norm = weight_norm[:, Q]
        loss = torch.sum(shuffled_weight_norm[P[idx1], :] * penalty[idx1, :] + shuffled_weight_norm[P[idx2], :] * penalty[idx2, :])
        loss_exchanged = torch.sum(shuffled_weight_norm[P[idx2], :] * penalty[idx1, :] + shuffled_weight_norm[P[idx1], :] * penalty[idx2, :])
    else:
        shuffled_weight_norm = weight_norm[P, :]
        loss = torch.sum(shuffled_weight_norm[:, Q[idx1]] * penalty[:, idx1] + shuffled_weight_norm[:, Q[idx2]] * penalty[:, idx2])
        loss_exchanged = torch.sum(shuffled_weight_norm[:, Q[idx2]] * penalty[:, idx1] + shuffled_weight_norm[:, Q[idx1]] * penalty[:, idx2])
    return True if loss_exchanged < loss else False

def update_one_module(module, iters, name):
    P, Q, weight, penalty = module.P.cpu().numpy(), module.Q.cpu().numpy(), module.weight.data.cpu().numpy(), module.penalty.cpu().numpy()
    weight_norm = np.sum(np.abs(weight), axis=(2,3))
    for i in range(iters):
        idx1, idx2 = np.random.choice(module.out_channels, size=2, replace=False)
        if compare_loss(P, Q, idx1, idx2, weight_norm, penalty, row=True):
            tmp = P[idx1]
            P[idx1] = P[idx2]
            P[idx2] = tmp
        idx1, idx2 = np.random.choice(module.in_channels, size=2, replace=False)
        if compare_loss(P, Q, idx1, idx2, weight_norm, penalty, row=False):
            tmp = Q[idx1]
            Q[idx1] = Q[idx2]
            Q[idx2] = tmp
    print("Iterate over later %s" % name)
    return torch.from_numpy(P), torch.from_numpy(Q)

def compare_loss(P, Q, idx1, idx2, weight_norm, penalty, row=True):
    if row:
        shuffled_weight_norm = weight_norm[:, Q]
        loss = np.sum(shuffled_weight_norm[P[idx1], :] * penalty[idx1, :] + shuffled_weight_norm[P[idx2], :] * penalty[idx2, :])
        loss_exchanged = np.sum(shuffled_weight_norm[P[idx2], :] * penalty[idx1, :] + shuffled_weight_norm[P[idx1], :] * penalty[idx2, :])
    else:
        shuffled_weight_norm = weight_norm[P, :]
        loss = np.sum(shuffled_weight_norm[:, Q[idx1]] * penalty[:, idx1] + shuffled_weight_norm[:, Q[idx2]] * penalty[:, idx2])
        loss_exchanged = np.sum(shuffled_weight_norm[:, Q[idx2]] * penalty[:, idx1] + shuffled_weight_norm[:, Q[idx1]] * penalty[:, idx2])
    return True if loss_exchanged < loss else False

@torch.no_grad()
def update_permutation_matrix(model, iters=1, mp=True):
    if mp:
        model.cpu()
        pool = multiprocessing.Pool(processes=20)
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
    else:
        for name, m in model.named_modules():
            if isinstance(m, GroupableConv2d):
                update_one_module_(m, iters)

@torch.no_grad()
def mask_group(model, factors, thres, logger):
    group_levels = {}
    total_connections = 0
    remaining_connections = 0
    for name, m in model.named_modules():
        if isinstance(m, GroupableConv2d):
            level = get_level(factors[name], thres)
            mask = m.mask_group(level)
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

if __name__ == "__main__":
    from resnet_cifar import resnet50
    import time
    model = resnet50(group1x1=True)
    start = time.time()
    update_permutation_matrix(model, iters=5000, mp=True)
    print(time.time() - start)

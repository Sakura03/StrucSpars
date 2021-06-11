import torch
import torch.distributed as dist
from groupconv import GroupableConv2d, get_cost_mat

@torch.no_grad()
def get_level(matrix, thres):
    matrix = matrix.clone()
    cost_mat = get_cost_mat(matrix.size(0), matrix.size(1))
    u = torch.unique(cost_mat, sorted=True)
    sums = [torch.sum(matrix).item()]
    for i in range(1, u.size(0)):
        mask = (cost_mat == u[-i])
        matrix[mask] = 0.0
        sums.append(torch.sum(matrix).item())
    percents = [s / (sums[0] + 1e-12) for s in sums]
    for level, s in enumerate(percents):
        if s < 1.0 - thres:
            break
    return level, percents


@torch.no_grad()
def get_struc_reg_mat(model):
    struc_reg_mat_dict = dict()
    for name, m in model.named_modules():
        if isinstance(m, GroupableConv2d):
            struc_reg_mat_dict[name] = m.struc_reg_mat[m.P, :][:, m.Q]
    return struc_reg_mat_dict


@torch.no_grad()
def get_perm_weight_norm(model):
    weight_norm_dict = dict()
    for name, m in model.named_modules():
        if isinstance(m, GroupableConv2d):
            weight_norm = torch.norm(m.weight.data.view(m.out_channels, m.in_channels, -1), dim=-1, p=1)
            weight_norm_dict[name] = weight_norm[m.P, :][:, m.Q]
    return weight_norm_dict


@torch.no_grad()
def get_sparsity(weight_norm_dict, thres):
    total0 = 0
    total = 0
    for name, wn in weight_norm_dict.items():
        # m=9 for conv3x3
        m = 9 if "conv2" in name else 1
        group_level, _ = get_level(wn, thres)
        total0 += wn.numel() * m / (2 ** (group_level - 1))
        total += wn.numel() * m
    return 1.0 - float(total0) / float(total)


@torch.no_grad()
def impose_group_lasso(model, l1lambda):
    for m in model.modules():
        if isinstance(m, GroupableConv2d):
            m.impose_struc_reg(l1lambda)


def get_threshold(model, target_sparsity, head=0., tail=1., margin=0.01, max_iters=10):
    weight_norm = get_perm_weight_norm(model)
    sparsity = get_sparsity(weight_norm, thres=(head+tail)/2)
    if (sparsity >= target_sparsity and sparsity-target_sparsity <= margin) or max_iters==0:
        # the ONLY output port
        return (head+tail)/2 if sparsity >= target_sparsity else tail
    else:
        if sparsity >= target_sparsity:
            return get_threshold(model, target_sparsity, head=head, tail=(head+tail)/2, max_iters=max_iters-1)
        else:
            return get_threshold(model, target_sparsity, head=(head+tail)/2, tail=tail, max_iters=max_iters-1)


@torch.no_grad()
def set_group_levels(model, group_levels):
    for name, m in model.named_modules():
        if isinstance(m, GroupableConv2d):
            m.set_group_level(group_levels[name])


@torch.no_grad()
def update_permutation_matrix(model, iters=1):
    print("Update permutation matrices for %d iters..." % iters)
    for m in model.modules():
        if isinstance(m, GroupableConv2d):
            m.update_PQ(iters)


@torch.no_grad()
def mask_group(model, factors, thres, logger=None):
    group_levels = {}
    for name, m in model.named_modules():
        if isinstance(m, GroupableConv2d):
            level, percents = get_level(factors[name], thres)
            group_levels[name] = level
            m.set_group_level(level)
            m.mask_group()
            if logger is not None:
                info = "Layer %s, weight size %s, group level %d, percents:" % (name, str(list(m.weight.size())), level)
                for p in percents:
                    info += " %.3f" % p
                logger.info(info)
    return group_levels


@torch.no_grad()
def real_group(model):
    for m in model.modules():
        if isinstance(m, GroupableConv2d):
            m.real_group()


@torch.no_grad()
def synchronize_model(model):
    for m in model.modules():
        if isinstance(m, GroupableConv2d):
            m.level_t = torch.tensor([m.group_level]).cuda()
            dist.broadcast(m.P, 0)
            dist.broadcast(m.Q, 0)
            dist.broadcast(m.P_inv, 0)
            dist.broadcast(m.Q_inv, 0)
            dist.broadcast(m.level_t, 0)
            dist.broadcast(m.struc_reg_mat, 0)

            m.group_level = m.level_t.cpu().item()
            del m.level_t

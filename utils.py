import time, torch
import torch.nn as nn
import torch.distributed as dist
from model import GroupableConv2d, get_penalty_matrix

@torch.no_grad()
def get_level(matrix, thres):
    matrix = matrix.clone()
    penalty = get_penalty_matrix(matrix.size(0), matrix.size(1))
    u = torch.unique(penalty, sorted=True)
    sums = [torch.sum(matrix).item()]
    for i in range(1, u.size(0)):
        mask = (penalty == u[-i])
        matrix[mask] = 0.
        sums.append(torch.sum(matrix).item())
    percents = [s / (sums[0]+1e-12) for s in sums]
    for level, s in enumerate(percents):
        if s < 1. - thres:
            break
    return level, percents

@torch.no_grad()
def get_penalties(model):
    penalties = {}
    for name, m in model.named_modules():
        if isinstance(m, GroupableConv2d):
            penalties[name] = m.shuffled_penalty[m.P, :][:, m.Q].squeeze().float()
    return penalties

@torch.no_grad()
def get_factors(model):
    factors = {}
    for name, m in model.named_modules():
        if isinstance(m, GroupableConv2d):
            weight_norm = torch.norm(m.weight.data.view(m.out_channels, m.in_channels, -1).float(), dim=-1, p=1)
            factors[name] = weight_norm[m.P, :][:, m.Q]
    return factors

@torch.no_grad()
def get_sparsity(factors, thres):
    total0 = 0
    total = 0
    for name, v in factors.items():
        m = 9 if "conv2" in name else 1
        total0 += v.numel() * m / (2 ** (get_level(v, thres)[0]-1))
        total += v.numel() * m
    return 1. - float(total0) / total

@torch.no_grad()
def get_sparsity_from_model(model, thres):
    return get_sparsity(get_factors(model), thres)

def get_sparsity_loss(model, enable_grad=False):
    with torch.set_grad_enabled(enable_grad):
        loss = 0
        for m in model.modules():
            if isinstance(m, GroupableConv2d):
                loss += m.compute_regularity()
        return loss

@torch.no_grad()
def impose_group_lasso(model, l1lambda):
    for m in model.modules():
        if isinstance(m, GroupableConv2d):
            m.impose_regularity(l1lambda)
            
def get_threshold(model, target_sparsity, head=0., tail=1., margin=0.01, max_iters=10):
    sparsity = get_sparsity_from_model(model, thres=(head+tail)/2)
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
    start = time.time()
    for name, m in model.named_modules():
        if isinstance(m, GroupableConv2d):
            m.update_PQ(iters)
    print("Update permutation matrices for %d iters, elapsed time: %.3f" % (iters, time.time()-start))

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
                info = "Layer %s, weight size %s, group level %d, percents:" % \
                        (name, str(list(m.weight.size())), level)
                for p in percents:
                    info += " %.3f" % p
                logger.info(info)
    return group_levels

@torch.no_grad()
def real_group(model):
    for name, m in model.named_modules():
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
            dist.broadcast(m.shuffled_penalty, 0)

            m.group_level = m.level_t.cpu().item()
            del m.level_t

@torch.no_grad()
def init_params(model):
    for m in model.modules():
        if isinstance(m, (torch.nn.Conv2d, GroupableConv2d)):
            torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(m, (torch.nn.BatchNorm2d, torch.nn.GroupNorm)):
            torch.nn.init.constant_(m.weight, 1.)
            torch.nn.init.constant_(m.bias, 0.)
            torch.nn.init.constant_(m.running_mean, 0.)
            torch.nn.init.constant_(m.running_var, 1.)
            m.num_batches_tracked.zero_()

@torch.no_grad()
def repermute_matrices(model, shuffle_type='learned'):
    assert shuffle_type.lower() in ['random', 'none', 'shufflenet', 'learned']
    if shuffle_type.lower() == 'learned': return
    for m in model.modules():
        if isinstance(m, GroupableConv2d):
            if shuffle_type.lower() == 'random':
                m.P = torch.randperm(len(m.P)).to(m.P.device)
                m.Q = torch.randperm(len(m.Q)).to(m.Q.device)
                m.P_inv, m.Q_inv = torch.argsort(m.P), torch.argsort(m.Q)
            elif shuffle_type.lower() == 'none':
                m.P = torch.arange(len(m.P)).to(m.P.device)
                m.Q = torch.arange(len(m.Q)).to(m.Q.device)
                m.P_inv, m.Q_inv = torch.argsort(m.P), torch.argsort(m.Q)
            elif shuffle_type.lower() == 'shufflenet':
                groups = 2 ** (m.group_level-1)
                m.P = torch.arange(len(m.P)).reshape(groups, -1).t().reshape(-1).to(m.P.device)
                m.Q = torch.arange(len(m.Q)).reshape(groups, -1).t().reshape(-1).to(m.Q.device)
            m.mask_group()

class DataIterator(object):
    def __init__(self, dataloader):
        self.dataloader = dataloader
        self.iterator = enumerate(self.dataloader)

    def next(self):
        try:
            _, data = next(self.iterator)
        except:
            self.dataloader.reset()
            self.iterator = enumerate(self.dataloader)
            _, data = next(self.iterator)
        return data[0]["data"], data[0]["label"].squeeze().cuda().long()

def get_parameters(model):
	group_no_weight_decay = []
	group_weight_decay = []
	for pname, p in model.named_parameters():
		if pname.find('weight') >= 0 and len(p.size()) > 1:
			print('with weight decay:', pname, p.size())
			group_weight_decay.append(p)
		else:
			print('without weight decay:', pname, p.size())
			group_no_weight_decay.append(p)
	assert len(list(model.parameters())) == len(group_weight_decay) + len(group_no_weight_decay)
	groups = [dict(params=group_weight_decay), dict(params=group_no_weight_decay, weight_decay=0.)]
	return groups

class CrossEntropyLabelSmooth(nn.Module):
	def __init__(self, num_classes, epsilon):
		super(CrossEntropyLabelSmooth, self).__init__()
		self.num_classes = num_classes
		self.epsilon = epsilon
		self.logsoftmax = nn.LogSoftmax(dim=1)

	def forward(self, inputs, targets):
		log_probs = self.logsoftmax(inputs)
		targets = torch.zeros_like(log_probs).scatter_(1, targets.unsqueeze(1), 1)
		targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
		loss = (-targets * log_probs).mean(0).sum()
		return loss

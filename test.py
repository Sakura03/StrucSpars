import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

power = 0.3

class doubly_stochstic_matrix(nn.Module):
    def __init__(self, weight, reg=.1, dtype=torch.double):
        super(doubly_stochstic_matrix, self).__init__()
        self.dim_P, self.dim_Q = weight.size(0), weight.size(1)
        self.reg, self.dtype = reg, dtype
        self.register_buffer("penalty", get_penalty_matrix(self.dim_P, self.dim_Q, power=power).to(self.dtype))
        
        self.register_buffer("P", torch.eye(self.dim_P, dtype=self.dtype))
        self.register_buffer("P_u", torch.ones(self.dim_P, dtype=self.dtype) / self.dim_P)
        self.register_buffer("P_v", torch.ones(self.dim_P, dtype=self.dtype) / self.dim_P)
        self.register_buffer("Q", torch.eye(self.dim_Q, dtype=self.dtype))
        self.register_buffer("Q_u", torch.ones(self.dim_Q, dtype=self.dtype) / self.dim_Q)
        self.register_buffer("Q_v", torch.ones(self.dim_Q, dtype=self.dtype) / self.dim_Q)

        self.sinkhorn_knopp(weight.to(self.dtype), iters=100)
    
    @torch.no_grad()   
    def sinkhorn_knopp(self, weight, iters=1000, verbose=False):
        for i in range(iters):
            # update P
            M = self.penalty.mm(self.Q.t()).mm(weight.t())
            K = torch.exp(-M / self.reg)
            
            for j in range(100):
                uprev, vprev = self.P_u, self.P_v
                    
                self.P_v = torch.einsum("ij,i->j", (K, self.P_u))
                self.P_v.reciprocal_()
                self.P_u = torch.einsum("ij,j->i", (K, self.P_v))
                self.P_u.reciprocal_()
                
                if torch.any(torch.isnan(self.P_u)) or torch.any(torch.isnan(self.P_v)) or torch.any(torch.isinf(self.P_u)) or torch.any(torch.isinf(self.P_v)):
                    print("Warning: numerical error!!!")
                    self.P_u = uprev
                    self.P_v = vprev
                    break

            self.P = torch.einsum("i,ij,j->ij", (self.P_u, K, self.P_v))
            err_P = torch.abs(torch.sum(self.P, dim=0) - 1.).sum().item()
            
            # update Q
            M = weight.t().mm(self.P.t()).mm(self.penalty)
            K = torch.exp(-M / self.reg)
            
            for j in range(100):
                uprev, vprev = self.Q_u, self.Q_v
                    
                self.Q_v = torch.einsum("ij,i->j", (K, self.Q_u))
                self.Q_v.reciprocal_()
                self.Q_u = torch.einsum("ij,j->i", (K, self.Q_v))
                self.Q_u.reciprocal_()
                
                if torch.any(torch.isnan(self.Q_u)) or torch.any(torch.isnan(self.Q_v)) or torch.any(torch.isinf(self.Q_u)) or torch.any(torch.isinf(self.Q_v)):
                    print("Warning: numerical error!!!")
                    self.Q_u = uprev
                    self.Q_v = vprev
                    break
            
            self.Q = torch.einsum("i,ij,j->ij", (self.Q_u, K, self.Q_v))
            err_Q = torch.abs(torch.sum(self.Q, dim=0) - 1.).sum().item()
            print("Error: %.8f, %.8f" % (err_P, err_Q))

        
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
    weight = torch.zeros(dim1, dim2)
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
def get_mask_from_level(level, dim1, dim2):
    mask = torch.ones(dim1, dim2).cuda()
    penalty = get_penalty_matrix(dim1, dim2)
    u = torch.unique(penalty, sorted=True)
    for i in range(1, level):
        mask[penalty == u[-i]] = 0.
    return mask

class GroupableConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, bias=True):
        super(GroupableConv2d, self).__init__(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                              stride=stride, padding=padding, dilation=dilation, bias=bias)
        self.permutations = doubly_stochstic_matrix(torch.norm(self.weight.view(self.out_channels, self.in_channels, -1), dim=-1, p=1))
        self.mask_grouped, self.real_grouped = False, False
    
    def compute_regularity(self):
        weight_norm = torch.norm(self.weight.view(self.out_channels, self.in_channels, -1), dim=-1, p=1)
        self.permutations.sinkhorn_knopp(weight_norm, iters=1)
        rearranged_norm = self.permutations.P.mm(self.weight).mm(self.permutations.Q)
        return torch.sum(rearranged_norm * self.permutations.penalty)
    
    @torch.no_grad()
    def mask_group(self, group_level):
        self.groupeded = True
        mask = get_mask_from_level(group_level, self.out_channels, self.in_channels)
        _, P_inv = torch.sort(self.P)
        _, Q_inv = torch.sort(self.Q)
        self.permuted_mask = mask[P_inv, :][:, Q_inv]
        self.permuted_mask.unsqueeze_(dim=-1).unsqueeze_(dim=-1)
        self.weight.data *= self.permuted_mask
        return self.permuted_mask
    
    @torch.no_grad()
    def real_group(self, group_level):
        self.real_grouped = True
        self.groups = 2 ** (group_level-1)
        weight = torch.zeros(self.out_channels, self.in_channels // self.groups, *self.kernel_size).to(self.weight.data.device)
        split_out, split_in = self.out_channels // self.groups, self.in_channels // self.groups
        for g in range(self.groups):
            permuted_weight = self.weight.data[self.P, :][:, self.Q]
            weight[g*split_out:(g+1)*split_out] = permuted_weight[g*split_out:(g+1)*split_out, g*split_in:(g+1)*split_in, :, :]
        self.weight = nn.Parameter(weight)
        _, self.P_inv = torch.sort(self.P)
        del self.penalty
    
    def forward(self, x):
        if self.real_grouped:
            x = x[:, self.Q, :, :]
            out = F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
            out = out[:, self.P_inv, :, :]
        elif self.mask_grouped:
            self.weight.data *= self.permuted_mask
            out = F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        else:
            out = F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        return out

dim1, dim2 = 512, 512
model = GroupableConv2d(dim1, dim2, 3, 1, 1, bias=False)
optimizer = torch.optim.SGD(model.parameters(), lr=0.1, weight_decay=0., momentum=0.)

for i in range(100):
    pass

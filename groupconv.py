import torch, ot
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair


@torch.no_grad()
def get_cost_mat(dim1, dim2, level=None, power=0.5):
    if level is None: level = 100
    struc_reg = torch.zeros(dim1, dim2)
    assign_location(struc_reg, 1.0, level, power)
    return struc_reg


@torch.no_grad()
def assign_location(tensor, value, level, power):
    dim1, dim2 = tensor.size()
    if dim1 % 2 == 0 and dim2 % 2 == 0 and level > 0:
        tensor[dim1//2:, :dim2//2] = value
        tensor[:dim1//2, dim2//2:] = value
        assign_location(tensor[dim1//2:, dim2//2:], value*power, level-1, power)
        assign_location(tensor[:dim1//2, :dim2//2], value*power, level-1, power)


@torch.no_grad()
def get_mask_from_level(level, dim1, dim2):
    mask = torch.ones(dim1, dim2)
    struc_reg = get_cost_mat(dim1, dim2, level-1)
    u = torch.unique(struc_reg, sorted=True)
    for i in range(1, level):
        mask[struc_reg == u[-i]] = 0.0
    return mask

    def _conv_forward(self, input: Tensor, weight: Tensor, bias: Optional[Tensor]):
        if self.padding_mode != 'zeros':
            return F.conv2d(F.pad(input, self._reversed_padding_repeated_twice, mode=self.padding_mode),
                            weight, bias, self.stride,
                            _pair(0), self.dilation, self.groups)
        return F.conv2d(input, weight, bias, self.stride,
                        self.padding, self.dilation, self.groups)

    def forward(self, input: Tensor) -> Tensor:
        return self._conv_forward(input, self.weight, self.bias)


class GroupableConv2d(nn.Conv2d):
    r"""
    attributes:
        a) group_level: current group level;
        b) power: decay factor of structured regularization matrix;
        c) mask_grouped/real_grouped: whether `mask_group`/`real_group` has been called;
        d) P/Q/P_inv/Q_inv: row/column permutation matrices and their inverses;
        e) struc_reg_mat: row- and column-shuffled structured regularization matrix (depending on P, Q, power, and group_level);
        f) cost_mat: cost matrix in linear programming;
        g) permutated_mask: row- and column-shuffled mask matrix after mask group.

    methods:
        a) set_group_level: set attributes --- group_level and struc_reg_mat;
        b) update_PQ: solve the linear programs and update P and Q alternatively (update variables: P, Q, P_inv, Q_inv, and struc_reg_mat);
        c) impose_struc_reg: directly imposes structured L1-refularization on the conv weights;
        d) mask_group: after calling, mask part of conv weights in each fwd pass;
        e) real_group: actually prunes conv weights and form GroupConvs.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, bias=True, power=0.5):
        kernel_size_ = _pair(kernel_size)
        stride_ = _pair(stride)
        padding_ = _pair(padding)
        dilation_ = _pair(dilation)
        super(GroupableConv2d, self).__init__(in_channels, out_channels, kernel_size_, stride_, padding_, dilation_, bias=bias)

        self.group_level = 1
        self.power = power
        self.mask_grouped, self.real_grouped = False, False
        self.register_buffer("P", torch.arange(self.out_channels))
        self.register_buffer("P_inv", torch.arange(self.out_channels))
        self.register_buffer("Q", torch.arange(self.in_channels))
        self.register_buffer("Q_inv", torch.arange(self.in_channels))
        struc_reg = get_cost_mat(self.out_channels, self.in_channels, level=self.group_level, power=self.power)
        self.register_buffer('struc_reg_mat', struc_reg)
        self.cost_mat = get_cost_mat(self.out_channels, self.in_channels, power=self.power).numpy().astype(np.float64)

    @staticmethod
    def matrix2idx(P, row=True):
        idx = np.argmax(P, axis=1 if row else 0)
        assert np.all(np.bincount(idx) == 1), "Invalid permutation matrix."
        return idx 

    def set_group_level(self, group_level):
        if self.group_level != group_level:
            self.group_level = group_level
            struc_reg = get_cost_mat(self.out_channels, self.in_channels, level=self.group_level, power=self.power).to(device=self.weight.data.device)
            self.struc_reg_mat = struc_reg[self.P_inv, :][:, self.Q_inv]

    def update_PQ(self, iters):
        ones_P, ones_Q = np.ones((self.out_channels, ), dtype=np.float64), np.ones((self.in_channels, ), dtype=np.float64)
        P, Q = self.P.cpu().numpy(), self.Q.cpu().numpy()
        weight = self.weight.data.cpu().numpy().astype(np.float64).reshape(self.out_channels, self.in_channels, -1)
        weight_norm = np.linalg.norm(weight, ord=1, axis=-1)

        for _ in range(iters):
            permutated_weight_norm = weight_norm[:, Q]
            M = np.matmul(self.cost_mat, permutated_weight_norm.T)
            P = ot.emd(ones_P, ones_P, M)
            P = self.matrix2idx(P, row=True)
            loss1 = np.sum(weight_norm[P, :][:, Q] * self.cost_mat)

            permutated_weight_norm = weight_norm[P, :]
            M = np.matmul(permutated_weight_norm.T, self.cost_mat)
            Q = ot.emd(ones_Q, ones_Q, M)
            Q = self.matrix2idx(Q, row=False)
            loss2 = np.sum(weight_norm[P, :][:, Q] * self.cost_mat)
            if loss1 == loss2: break

        self.P, self.Q = torch.from_numpy(P).to(device=self.weight.data.device), torch.from_numpy(Q).to(device=self.weight.data.device)
        self.P_inv, self.Q_inv = torch.argsort(self.P), torch.argsort(self.Q)
        struc_reg = get_cost_mat(self.out_channels, self.in_channels, level=self.group_level, power=self.power).to(device=self.weight.data.device)
        self.struc_reg_mat = struc_reg[self.P_inv, :][:, self.Q_inv]

    def impose_struc_reg(self, l1lambda):
        self.weight.grad.add_(l1lambda * (torch.sign(self.weight.data) * self.struc_reg_mat.unsqueeze(-1).unsqueeze(-1)))

    def mask_group(self):
        mask = get_mask_from_level(self.group_level, self.out_channels, self.in_channels).to(device=self.weight.data.device)
        self.permuted_mask = mask[self.P_inv, :][:, self.Q_inv].unsqueeze(-1).unsqueeze(-1)
        self.weight.data *= self.permuted_mask
        if hasattr(self, "cost_mat"): del self.cost_mat
        if hasattr(self, "struc_reg_mat"): del self.struc_reg_mat
        self.mask_grouped = True

    def real_group(self):
        self.groups = 2 ** (self.group_level - 1)
        weight = torch.zeros(self.out_channels, self.in_channels // self.groups, *self.kernel_size).to(device=self.weight.data.device)
        split_out, split_in = self.out_channels // self.groups, self.in_channels // self.groups
        for g in range(self.groups):
            permuted_weight = self.weight.data[self.P, :][:, self.Q]
            weight[g*split_out:(g+1)*split_out] = permuted_weight[g*split_out:(g+1)*split_out, g*split_in:(g+1)*split_in, :, :]
        del self.weight
        self.weight = nn.Parameter(weight)
        if hasattr(self, "cost_mat"): del self.cost_mat
        if hasattr(self, "struc_reg_mat"): del self.struc_reg_mat
        self.real_grouped = True

    def forward(self, x):
        if self.real_grouped:
            x = x[:, self.Q, :, :]
        elif self.mask_grouped:
            self.weight.data *= self.permuted_mask
        out = F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        if self.real_grouped:
            out = out[:, self.P_inv, :, :]
        return out

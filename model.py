import torch, ot, re
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torch.utils.checkpoint as cp
from collections import OrderedDict
__all__ = ['ResNet', 'ResNet_cifar', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152', 'resnext50_32x4d', 'resnext101_32x8d', 'wide_resnet50_2',
           'wide_resnet101_2', 'resnet20_cifar', 'resnet32_cifar', 'resnet44_cifar',
           'resnet56_cifar', 'resnet110_cifar', 'resnet1202_cifar']

#####################################################################################
#####################################################################################
#### model template of GroupableConv2d
#####################################################################################
#####################################################################################

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
def get_penalty_matrix(dim1, dim2, level=None, power=0.5):
    assert isPower(dim1) and isPower(dim2)
    if level is None: level = 100
    weight = torch.zeros(dim1, dim2)
    assign_location(weight, 1., level, power)
    return weight

@torch.no_grad()
def assign_location(tensor, num, level, power):
    dim1, dim2 = tensor.size()
    if dim1 == 1 or dim2 == 1 or level == 0:
        return
    else:
        tensor[dim1//2:, :dim2//2] = num
        tensor[:dim1//2, dim2//2:] = num
        assign_location(tensor[dim1//2:, dim2//2:], num*power, level-1, power)
        assign_location(tensor[:dim1//2, :dim2//2], num*power, level-1, power)

@torch.no_grad()
def get_mask_from_level(level, dim1, dim2):
    mask = torch.ones(dim1, dim2)
    penalty = get_penalty_matrix(dim1, dim2, level-1)
    u = torch.unique(penalty, sorted=True)
    for i in range(1, level):
        mask[penalty == u[-i]] = 0.
    return mask

class GroupableConv2d(nn.Conv2d):
    r"""
    attributes of `GroupableConv2d`:
        a) `group_level` indicates the number of groups that can be achieved currently;
        b) `power` indicates the decay power of the attribute `template`;
        c) `mask_grouped` and `real_grouped` indicate whether `mask_group` or `real_group` has been called, respectively;
        d) `P`, `Q`, `P_inv`, `Q_inv` indicate the row and column permutation matrices and their inverses, respectively;
        e) `shuffled_penalty` indicates the row- and column-shuffled penalty matrix (which depends on `P`, `Q`, `power`, and `group_level`) (shape: [C_out, C_in , 1, 1]);
        f) `template` indicates the penalty matrix when computing the optimal permutations `P` and `Q`;
        g) `permutated_mask` indicates the row- and column-shuffled mask matrix after `mask_group`.
    
    methods of `GroupableConv2d`:
        a) `set_group_level` receives the current group level, and set the current `group_level` and `shuffled_penalty` accordingly;
        b) `update_PQ` receives the number of iterations, and update the permutations `P`, `Q`, `P_inv`, `Q_inv`, and `shuffled_penalty`;
        c) `compute_regularity` computes the sparsity regularity according to the current `shuffled_penalty`;
        d) `impose_regularity` imposes L1-penalty on the conv weights directly, according to the current `shuffled_penalty`;
        e) `mask_group` creates `permutated_mask` and mask the conv weights according to the current `group_level` attribute;
        f) `real_group` actually sets the attribute `groups`, and prunes the conv weights
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, bias=True, power=0.5):
        super(GroupableConv2d, self).__init__(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                              stride=stride, padding=padding, dilation=dilation, bias=bias)
        self.group_level = 1
        self.power = power
        self.mask_grouped, self.real_grouped = False, False
        self.register_buffer("P", torch.arange(self.out_channels))
        self.register_buffer("P_inv", torch.arange(self.out_channels))
        self.register_buffer("Q", torch.arange(self.in_channels))
        self.register_buffer("Q_inv", torch.arange(self.in_channels))
        penalty = get_penalty_matrix(self.out_channels, self.in_channels, level=self.group_level, power=self.power)
        self.register_buffer('shuffled_penalty', penalty[self.P_inv, :][:, self.Q_inv])
        # self.shuffled_penalty = penalty[self.P_inv, :][:, self.Q_inv]
        self.shuffled_penalty.unsqueeze_(-1).unsqueeze_(-1)
        self.template = get_penalty_matrix(self.out_channels, self.in_channels, power=self.power).numpy().astype(np.float64)

    def matrix2idx(self, P, row=True):
        idx = np.argmax(P, axis=1 if row else 0)
        assert np.all(np.bincount(idx) == 1), "Invalid permutation matrix."
        return idx 
    
    def set_group_level(self, group_level):
        if self.group_level != group_level:
            self.group_level = group_level
            penalty = get_penalty_matrix(self.out_channels, self.in_channels, level=self.group_level, power=self.power).to(device=self.weight.data.device, dtype=self.weight.data.dtype)
            self.shuffled_penalty = penalty[self.P_inv, :][:, self.Q_inv]
            self.shuffled_penalty.unsqueeze_(-1).unsqueeze_(-1)
    
    def update_PQ(self, iters):
        ones_P, ones_Q = np.ones((self.out_channels, ), dtype=np.float64), np.ones((self.in_channels, ), dtype=np.float64)
        P, Q = self.P.cpu().numpy(), self.Q.cpu().numpy()
        weight = self.weight.data.cpu().numpy().astype(np.float64).reshape(self.out_channels, self.in_channels, -1)
        weight_norm = np.linalg.norm(weight, ord=1, axis=-1)

        for _ in range(iters):
            permutated_weight_norm = weight_norm[:, Q]
            M = np.matmul(self.template, permutated_weight_norm.T)
            P = ot.emd(ones_P, ones_P, M)
            P = self.matrix2idx(P, row=True)
            loss1 = np.sum(weight_norm[P, :][:, Q] * self.template)
            
            permutated_weight_norm = weight_norm[P, :]
            M = np.matmul(permutated_weight_norm.T, self.template)
            Q = ot.emd(ones_Q, ones_Q, M)
            Q = self.matrix2idx(Q, row=False)
            loss2 = np.sum(weight_norm[P, :][:, Q] * self.template)
            if loss1 == loss2: break
        
        self.P, self.Q = torch.from_numpy(P).to(device=self.weight.data.device), torch.from_numpy(Q).to(device=self.weight.data.device)
        self.P_inv, self.Q_inv = torch.argsort(self.P), torch.argsort(self.Q)
        penalty = get_penalty_matrix(self.out_channels, self.in_channels, level=self.group_level, power=self.power).to(device=self.weight.data.device, dtype=self.weight.data.dtype)
        self.shuffled_penalty = penalty[self.P_inv, :][:, self.Q_inv]
        self.shuffled_penalty.unsqueeze_(-1).unsqueeze_(-1)
    
    def compute_regularity(self):
        weight_norm = torch.norm(self.weight.view(self.out_channels, self.in_channels, -1), dim=-1, p=1)
        return torch.sum(weight_norm * self.shuffled_penalty.squeeze())
    
    def impose_regularity(self, l1lambda):
        self.weight.grad.add_(l1lambda * (torch.sign(self.weight.data) * self.shuffled_penalty))
    
    def mask_group(self):
        self.mask_grouped = True
        mask = get_mask_from_level(self.group_level, self.out_channels, self.in_channels).to(device=self.weight.data.device, dtype=self.weight.data.dtype)
        self.permuted_mask = mask[self.P_inv, :][:, self.Q_inv]
        self.permuted_mask.unsqueeze_(dim=-1).unsqueeze_(dim=-1)
        self.weight.data *= self.permuted_mask
        if hasattr(self, "template"): del self.template
        if hasattr(self, "shuffled_penalty"): del self.shuffled_penalty

    def real_group(self):
        self.real_grouped = True
        self.groups = 2 ** (self.group_level-1)
        weight = torch.zeros(self.out_channels, self.in_channels // self.groups, *self.kernel_size).to(device=self.weight.data.device, dtype=self.weight.data.dtype)
        split_out, split_in = self.out_channels // self.groups, self.in_channels // self.groups
        for g in range(self.groups):
            permuted_weight = self.weight.data[self.P, :][:, self.Q]
            weight[g*split_out:(g+1)*split_out] = permuted_weight[g*split_out:(g+1)*split_out, g*split_in:(g+1)*split_in, :, :]
        del self.weight
        self.weight = nn.Parameter(weight)
        if hasattr(self, "template"): del self.template
        if hasattr(self, "shuffled_penalty"): del self.shuffled_penalty

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

#####################################################################################
#####################################################################################
#### ResNet model for ImageNet
#####################################################################################
#####################################################################################

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1, groupable=True, power=0.5):
    """3x3 convolution with padding"""
    return GroupableConv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=dilation, bias=False, dilation=dilation, power=power) if groupable \
           else nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=dilation, groups=groups, bias=False, dilation=dilation)

def conv1x1(in_planes, out_planes, stride=1, groupable=False, power=0.5):
    """1x1 convolution"""
    return GroupableConv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False, power=power) if groupable \
           else nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, group1x1=False,
                 group3x3=True, power=0.5):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride, groupable=group3x3, power=power)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, groupable=group3x3, power=power)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, group1x1=False,
                 group3x3=True, power=0.5):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width, groupable=group1x1, power=power)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation, groupable=group3x3, power=power)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion, groupable=group1x1, power=power)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None, group1x1=False, group3x3=True, power=0.5):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        self.power = power
        self.group1x1 = group1x1
        self.group3x3 = group3x3
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, (nn.Conv2d, GroupableConv2d)):
                init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride, groupable=self.group1x1, power=self.power),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer,
                            group1x1=self.group1x1, group3x3=self.group3x3, power=self.power))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer, group1x1=self.group1x1,
                                group3x3=self.group3x3, power=self.power))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x


def _resnet(arch, block, layers, pretrained, progress, **kwargs):
    model = ResNet(block, layers, **kwargs)
    return model


def resnet18(**kwargs):
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    """
    return _resnet('resnet18', BasicBlock, [2, 2, 2, 2], **kwargs)


def resnet34(**kwargs):
    r"""ResNet-34 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    """
    return _resnet('resnet34', BasicBlock, [3, 4, 6, 3], **kwargs)


def resnet50(**kwargs):
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    """
    return _resnet('resnet50', Bottleneck, [3, 4, 6, 3], **kwargs)


def resnet101(**kwargs):
    r"""ResNet-101 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    """
    return _resnet('resnet101', Bottleneck, [3, 4, 23, 3], **kwargs)


def resnet152(**kwargs):
    r"""ResNet-152 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    """
    return _resnet('resnet152', Bottleneck, [3, 8, 36, 3], **kwargs)


def resnext50_32x4d(**kwargs):
    r"""ResNeXt-50 32x4d model from
    `"Aggregated Residual Transformation for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_
    """
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 4
    return _resnet('resnext50_32x4d', Bottleneck, [3, 4, 6, 3], **kwargs)


def resnext101_32x8d(**kwargs):
    r"""ResNeXt-101 32x8d model from
    `"Aggregated Residual Transformation for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_
    """
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 8
    return _resnet('resnext101_32x8d', Bottleneck, [3, 4, 23, 3], **kwargs)


def wide_resnet50_2(**kwargs):
    r"""Wide ResNet-50-2 model from
    `"Wide Residual Networks" <https://arxiv.org/pdf/1605.07146.pdf>`_

    The model is the same as ResNet except for the bottleneck number of channels
    which is twice larger in every block. The number of channels in outer 1x1
    convolutions is the same, e.g. last block in ResNet-50 has 2048-512-2048
    channels, and in Wide ResNet-50-2 has 2048-1024-2048.
    """
    kwargs['width_per_group'] = 64 * 2
    return _resnet('wide_resnet50_2', Bottleneck, [3, 4, 6, 3], **kwargs)


def wide_resnet101_2(**kwargs):
    r"""Wide ResNet-101-2 model from
    `"Wide Residual Networks" <https://arxiv.org/pdf/1605.07146.pdf>`_

    The model is the same as ResNet except for the bottleneck number of channels
    which is twice larger in every block. The number of channels in outer 1x1
    convolutions is the same, e.g. last block in ResNet-50 has 2048-512-2048
    channels, and in Wide ResNet-50-2 has 2048-1024-2048.
    """
    kwargs['width_per_group'] = 64 * 2
    return _resnet('wide_resnet101_2', Bottleneck, [3, 4, 23, 3], **kwargs)

#####################################################################################
#####################################################################################
#### ResNet model for CIFAR
#####################################################################################
#####################################################################################

def _weights_init(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)

class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)

class BasicBlock_cifar(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, groupable=True, power=0.5):
        super(BasicBlock_cifar, self).__init__()
        self.conv1 = GroupableConv2d(in_planes, planes, 3, stride, 1, bias=False, power=power) if groupable \
                     else nn.Conv2d(in_planes, planes, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = GroupableConv2d(planes, planes, 3, 1, 1, bias=False, power=power) if groupable \
                     else nn.Conv2d(planes, planes, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                GroupableConv2d(in_planes, self.expansion*planes, 1, stride, bias=False, power=power) if groupable \
                else nn.Conv2d(in_planes, self.expansion*planes, 1, stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out


class ResNet_cifar(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, groupable=True, power=0.5):
        super(ResNet_cifar, self).__init__()
        self.in_planes = 16
        self.groupable = groupable
        self.power = power
        self.conv1 = nn.Conv2d(3, 16, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.linear = nn.Linear(64, num_classes)

        self.apply(_weights_init)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, self.groupable, self.power))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def resnet20_cifar(**kwargs):
    return ResNet_cifar(BasicBlock_cifar, [3, 3, 3], **kwargs)


def resnet32_cifar(**kwargs):
    return ResNet_cifar(BasicBlock_cifar, [5, 5, 5], **kwargs)


def resnet44_cifar(**kwargs):
    return ResNet_cifar(BasicBlock_cifar, [7, 7, 7], **kwargs)


def resnet56_cifar(**kwargs):
    return ResNet_cifar(BasicBlock_cifar, [9, 9, 9], **kwargs)


def resnet110_cifar(**kwargs):
    return ResNet_cifar(BasicBlock_cifar, [18, 18, 18], **kwargs)


def resnet1202_cifar(**kwargs):
    return ResNet_cifar(BasicBlock_cifar, [200, 200, 200], **kwargs)

#####################################################################################
#####################################################################################
#### VGGNet model for ImageNet
#####################################################################################
#####################################################################################

class VGG(nn.Module):

    def __init__(self, features, num_classes=1000, init_weights=True):
        super(VGG, self).__init__()
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, GroupableConv2d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


def make_layers(cfg, batch_norm=False, groupable=True):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = GroupableConv2d(in_channels, v, 3, 1, 1) if groupable else nn.Conv2d(in_channels, v, 3, 1, 1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


cfgs = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


def _vgg(arch, cfg, batch_norm, groupable, **kwargs):
    model = VGG(make_layers(cfgs[cfg], batch_norm=batch_norm, groupable=groupable), **kwargs)
    return model


def vgg11(**kwargs):
    r"""VGG 11-layer model (configuration "A") from
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_
    """
    return _vgg('vgg11', 'A', False, **kwargs)


def vgg11_bn(**kwargs):
    r"""VGG 11-layer model (configuration "A") with batch normalization
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_
    """
    return _vgg('vgg11_bn', 'A', True, **kwargs)


def vgg13(**kwargs):
    r"""VGG 13-layer model (configuration "B")
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_
    """
    return _vgg('vgg13', 'B', False, **kwargs)


def vgg13_bn(**kwargs):
    r"""VGG 13-layer model (configuration "B") with batch normalization
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_
    """
    return _vgg('vgg13_bn', 'B', True, **kwargs)


def vgg16(**kwargs):
    r"""VGG 16-layer model (configuration "D")
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_
    """
    return _vgg('vgg16', 'D', False, **kwargs)


def vgg16_bn(**kwargs):
    r"""VGG 16-layer model (configuration "D") with batch normalization
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_
    """
    return _vgg('vgg16_bn', 'D', True, **kwargs)


def vgg19(**kwargs):
    r"""VGG 19-layer model (configuration "E")
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_
    """
    return _vgg('vgg19', 'E', False, **kwargs)


def vgg19_bn(**kwargs):
    r"""VGG 19-layer model (configuration 'E') with batch normalization
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_
    """
    return _vgg('vgg19_bn', 'E', True, **kwargs)

#####################################################################################
#####################################################################################
#### DenseNet model for ImageNet
#####################################################################################
#####################################################################################

def _bn_function_factory(norm, relu, conv):
    def bn_function(*inputs):
        concated_features = torch.cat(inputs, 1)
        bottleneck_output = conv(relu(norm(concated_features)))
        return bottleneck_output

    return bn_function


class _DenseLayer(nn.Sequential):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate, groupable, memory_efficient=False):
        super(_DenseLayer, self).__init__()
        self.add_module('norm1', nn.BatchNorm2d(num_input_features)),
        self.add_module('relu1', nn.ReLU(inplace=True)),
        self.add_module('conv1', GroupableConv2d(num_input_features, bn_size * growth_rate, 1, 1, 0, bias=False) if groupable \
                                 else nn.Conv2d(num_input_features, bn_size * growth_rate, 1, 1, 0, bias=False)),
        self.add_module('norm2', nn.BatchNorm2d(bn_size * growth_rate)),
        self.add_module('relu2', nn.ReLU(inplace=True)),
        self.add_module('conv2', GroupableConv2d(bn_size * growth_rate, growth_rate, 3, 1, 1, bias=False) if groupable \
                                 else nn.Conv2d(bn_size * growth_rate, growth_rate, 3, 1, 1, bias=False)),
        self.drop_rate = drop_rate
        self.memory_efficient = memory_efficient

    def forward(self, *prev_features):
        bn_function = _bn_function_factory(self.norm1, self.relu1, self.conv1)
        if self.memory_efficient and any(prev_feature.requires_grad for prev_feature in prev_features):
            bottleneck_output = cp.checkpoint(bn_function, *prev_features)
        else:
            bottleneck_output = bn_function(*prev_features)
        new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate,
                                     training=self.training)
        return new_features


class _DenseBlock(nn.Module):
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate, groupable, memory_efficient=False):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(
                num_input_features + i * growth_rate,
                growth_rate=growth_rate,
                bn_size=bn_size,
                drop_rate=drop_rate,
                groupable=groupable,
                memory_efficient=memory_efficient,
            )
            self.add_module('denselayer%d' % (i + 1), layer)

    def forward(self, init_features):
        features = [init_features]
        for name, layer in self.named_children():
            new_features = layer(*features)
            features.append(new_features)
        return torch.cat(features, 1)


class _Transition(nn.Sequential):
    def __init__(self, num_input_features, num_output_features, groupable):
        super(_Transition, self).__init__()
        self.add_module('norm', nn.BatchNorm2d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', GroupableConv2d(num_input_features, num_output_features, 1, 1, 0, bias=False) if groupable \
                                else nn.Conv2d(num_input_features, num_output_features, 1, 1, 0, bias=False))
        self.add_module('pool', nn.AvgPool2d(kernel_size=2, stride=2))


class DenseNet(nn.Module):
    r"""Densenet-BC model class, based on
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_

    Args:
        growth_rate (int) - how many filters to add each layer (`k` in paper)
        block_config (list of 4 ints) - how many layers in each pooling block
        num_init_features (int) - the number of filters to learn in the first convolution layer
        bn_size (int) - multiplicative factor for number of bottle neck layers
          (i.e. bn_size * k features in the bottleneck layer)
        drop_rate (float) - dropout rate after each dense layer
        num_classes (int) - number of classification classes
        memory_efficient (bool) - If True, uses checkpointing. Much more memory efficient,
          but slower. Default: *False*. See `"paper" <https://arxiv.org/pdf/1707.06990.pdf>`_
    """

    def __init__(self, growth_rate=32, block_config=(6, 12, 24, 16),
                 num_init_features=64, bn_size=4, drop_rate=0, num_classes=1000,
                 groupable=True, memory_efficient=False):

        super(DenseNet, self).__init__()

        # First convolution
        self.features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(3, num_init_features, kernel_size=7, stride=2,
                                padding=3, bias=False)),
            ('norm0', nn.BatchNorm2d(num_init_features)),
            ('relu0', nn.ReLU(inplace=True)),
            ('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
        ]))

        # Each denseblock
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(
                num_layers=num_layers,
                num_input_features=num_features,
                bn_size=bn_size,
                growth_rate=growth_rate,
                drop_rate=drop_rate,
                groupable=groupable,
                memory_efficient=memory_efficient
            )
            self.features.add_module('denseblock%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = _Transition(num_input_features=num_features,
                                    num_output_features=num_features // 2,
                                    groupable=groupable)
                self.features.add_module('transition%d' % (i + 1), trans)
                num_features = num_features // 2

        # Final batch norm
        self.features.add_module('norm5', nn.BatchNorm2d(num_features))

        # Linear layer
        self.classifier = nn.Linear(num_features, num_classes)

        # Official init from torch repo.
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, GroupableConv2d)):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        features = self.features(x)
        out = F.relu(features, inplace=True)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)
        out = self.classifier(out)
        return out


def _densenet(arch, growth_rate, block_config, num_init_features, **kwargs):
    model = DenseNet(growth_rate, block_config, num_init_features, **kwargs)
    return model


def densenet121(**kwargs):
    r"""Densenet-121 model from
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
        memory_efficient (bool) - If True, uses checkpointing. Much more memory efficient,
          but slower. Default: *False*. See `"paper" <https://arxiv.org/pdf/1707.06990.pdf>`_
    """
    return _densenet('densenet121', 32, (6, 12, 24, 16), 64, **kwargs)


def densenet161(**kwargs):
    r"""Densenet-161 model from
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
        memory_efficient (bool) - If True, uses checkpointing. Much more memory efficient,
          but slower. Default: *False*. See `"paper" <https://arxiv.org/pdf/1707.06990.pdf>`_
    """
    return _densenet('densenet161', 48, (6, 12, 36, 24), 96, **kwargs)


def densenet169(**kwargs):
    r"""Densenet-169 model from
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
        memory_efficient (bool) - If True, uses checkpointing. Much more memory efficient,
          but slower. Default: *False*. See `"paper" <https://arxiv.org/pdf/1707.06990.pdf>`_
    """
    return _densenet('densenet169', 32, (6, 12, 32, 32), 64, **kwargs)


def densenet201(**kwargs):
    r"""Densenet-201 model from
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
        memory_efficient (bool) - If True, uses checkpointing. Much more memory efficient,
          but slower. Default: *False*. See `"paper" <https://arxiv.org/pdf/1707.06990.pdf>`_
    """
    return _densenet('densenet201', 32, (6, 12, 48, 32), 64, **kwargs)

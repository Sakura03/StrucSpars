import torch, ot
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torch.utils.checkpoint as cp
from collections import OrderedDict

__all__ = ['GroupableConv2d', 'resnet18', 'resnet34', 'resnet50', 'resnet101', 
           'resnet152', 'resnext50_32x4d', 'resnext101_32x8d', 'wide_resnet50_2',
           'wide_resnet101_2', 'resnet20_cifar', 'resnet32_cifar', 'resnet44_cifar',
           'resnet56_cifar', 'resnet110_cifar', 'resnet1202_cifar', 'densenet121',
           'densenet161', 'densenet169', 'densenet201']


#####################################################################################
#####################################################################################
#### model template of GroupableConv2d
#####################################################################################
#####################################################################################

@torch.no_grad()
def get_struc_reg_mat(dim1, dim2, level=None, power=0.5):
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
    struc_reg = get_struc_reg_mat(dim1, dim2, level-1)
    u = torch.unique(struc_reg, sorted=True)
    for i in range(1, level):
        mask[struc_reg == u[-i]] = 0.0
    return mask


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
        super(GroupableConv2d, self).__init__(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                              stride=stride, padding=padding, dilation=dilation, bias=bias)
        self.group_level = 1
        self.power = power
        self.mask_grouped, self.real_grouped = False, False
        self.register_buffer("P", torch.arange(self.out_channels))
        self.register_buffer("P_inv", torch.arange(self.out_channels))
        self.register_buffer("Q", torch.arange(self.in_channels))
        self.register_buffer("Q_inv", torch.arange(self.in_channels))
        struc_reg = get_struc_reg_mat(self.out_channels, self.in_channels, level=self.group_level, power=self.power)
        self.register_buffer('struc_reg_mat', struc_reg)
        self.cost_mat = get_struc_reg_mat(self.out_channels, self.in_channels, power=self.power).numpy().astype(np.float64)

    @staticmethod
    def matrix2idx(P, row=True):
        idx = np.argmax(P, axis=1 if row else 0)
        assert np.all(np.bincount(idx) == 1), "Invalid permutation matrix."
        return idx 

    def set_group_level(self, group_level):
        if self.group_level != group_level:
            self.group_level = group_level
            struc_reg = get_struc_reg_mat(self.out_channels, self.in_channels, level=self.group_level, power=self.power).to(device=self.weight.data.device)
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
        struc_reg = get_struc_reg_mat(self.out_channels, self.in_channels, level=self.group_level, power=self.power).to(device=self.weight.data.device)
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


#####################################################################################
#####################################################################################
#### ResNet model for ImageNet
#####################################################################################
#####################################################################################

def conv3x3(in_planes, out_planes, stride=1, dilation=1, power=0.5):
    """3x3 convolution with padding"""
    return GroupableConv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=dilation, bias=False, dilation=dilation, power=power)


def conv1x1(in_planes, out_planes, stride=1, power=0.5):
    """1x1 convolution"""
    return GroupableConv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False, power=power)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, dilation=1, norm_layer=None, power=0.5):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride, power=power)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, power=power)
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

    def __init__(self, inplanes, planes, stride=1, downsample=None, dilation=1, norm_layer=None, power=0.5):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, planes, power=power)
        self.bn1 = norm_layer(planes)
        self.conv2 = conv3x3(planes, planes, stride, dilation, power=power)
        self.bn2 = norm_layer(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion, power=power)
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
                 norm_layer=None, power=0.5):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        self.power = power
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
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

    def _make_layer(self, block, planes, blocks, stride=1):
        norm_layer = self._norm_layer
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride, power=self.power),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, norm_layer, power=self.power))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=self.dilation, norm_layer=norm_layer, power=self.power))

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


def _resnet(block, layers, **kwargs):
    model = ResNet(block, layers, **kwargs)
    return model


def resnet50(**kwargs):
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    """
    return _resnet(Bottleneck, [3, 4, 6, 3], **kwargs)


def resnet101(**kwargs):
    r"""ResNet-101 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    """
    return _resnet(Bottleneck, [3, 4, 23, 3], **kwargs)


def resnet152(**kwargs):
    r"""ResNet-152 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    """
    return _resnet(Bottleneck, [3, 8, 36, 3], **kwargs)


#####################################################################################
#####################################################################################
#### ResNet model for CIFAR
#####################################################################################
#####################################################################################

def _weights_init(m):
    if isinstance(m, nn.Linear) or isinstance(m, (nn.Conv2d, GroupableConv2d)):
        init.kaiming_normal_(m.weight)


class BasicBlockCIFAR(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, power=0.5):
        super(BasicBlockCIFAR, self).__init__()
        self.conv1 = GroupableConv2d(in_planes, planes, 3, stride, 1, bias=False, power=power)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = GroupableConv2d(planes, planes, 3, 1, 1, bias=False, power=power)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                GroupableConv2d(in_planes, self.expansion*planes, 1, stride, bias=False, power=power),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out


class ResNetCIFAR(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, power=0.5):
        super(ResNetCIFAR, self).__init__()
        self.in_planes = 16
        self.power = power
        self.conv1 = nn.Conv2d(3, 16, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(64, num_classes)

        self.apply(_weights_init)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, self.power))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.linear(out)
        return out


def resnet20_cifar(**kwargs):
    return ResNetCIFAR(BasicBlockCIFAR, [3, 3, 3], **kwargs)


def resnet32_cifar(**kwargs):
    return ResNetCIFAR(BasicBlockCIFAR, [5, 5, 5], **kwargs)


def resnet44_cifar(**kwargs):
    return ResNetCIFAR(BasicBlockCIFAR, [7, 7, 7], **kwargs)


def resnet56_cifar(**kwargs):
    return ResNetCIFAR(BasicBlockCIFAR, [9, 9, 9], **kwargs)


def resnet110_cifar(**kwargs):
    return ResNetCIFAR(BasicBlockCIFAR, [18, 18, 18], **kwargs)


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
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate, groupable, power, memory_efficient=False):
        super(_DenseLayer, self).__init__()
        self.add_module('norm1', nn.BatchNorm2d(num_input_features)),
        self.add_module('relu1', nn.ReLU(inplace=True)),
        self.add_module('conv1', GroupableConv2d(num_input_features, bn_size * growth_rate, 1, 1, 0, bias=False, power=power) if groupable \
                                 else nn.Conv2d(num_input_features, bn_size * growth_rate, 1, 1, 0, bias=False)),
        self.add_module('norm2', nn.BatchNorm2d(bn_size * growth_rate)),
        self.add_module('relu2', nn.ReLU(inplace=True)),
        self.add_module('conv2', GroupableConv2d(bn_size * growth_rate, growth_rate, 3, 1, 1, bias=False, power=power) if groupable \
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
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate, groupable, power, memory_efficient=False):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(
                num_input_features + i * growth_rate,
                growth_rate=growth_rate,
                bn_size=bn_size,
                drop_rate=drop_rate,
                groupable=groupable,
                power=power,
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
    def __init__(self, num_input_features, num_output_features, groupable, power):
        super(_Transition, self).__init__()
        self.add_module('norm', nn.BatchNorm2d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', GroupableConv2d(num_input_features, num_output_features, 1, 1, 0, bias=False, power=power) if groupable \
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
                 groupable=True, power=0.5, memory_efficient=False):

        super(DenseNet, self).__init__()
        # First convolution
        self.features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(3, num_init_features, kernel_size=7, stride=2,
                                padding=3, bias=False)),
            ('norm0', nn.BatchNorm2d(num_init_features)),
            ('relu0', nn.ReLU(inplace=True)),
            ('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
        ]))
        self.relu = nn.ReLU(inplace=True)

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
                power=power,
                memory_efficient=memory_efficient
            )
            self.features.add_module('denseblock%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = _Transition(num_input_features=num_features,
                                    num_output_features=num_features // 2,
                                    groupable=groupable, power=power)
                self.features.add_module('transition%d' % (i + 1), trans)
                num_features = num_features // 2

        # Final batch norm
        self.features.add_module('norm5', nn.BatchNorm2d(num_features))
        
        # Global average pooling
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

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
        out = self.relu(features, inplace=True)
        out = self.avgpool(out)
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

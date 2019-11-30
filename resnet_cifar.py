import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152']

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
        self.register_buffer("P", torch.arange(self.out_channels))
        self.register_buffer("Q", torch.arange(self.in_channels))
        self.register_buffer("penalty", get_penalty_matrix(self.out_channels, self.in_channels, 0.3))
        self.mask_grouped, self.real_grouped = False, False
        
    def compute_weight_norm(self, p=1):
        self.weight_norm = torch.norm(self.weight.data.view(self.out_channels, self.in_channels, -1), dim=-1, p=p)
    
    @torch.no_grad()
    def compare_loss(self, idx1, idx2, row=True):
        if row:
            shuffled_weight_norm = self.weight_norm[:, self.Q]
            loss = torch.sum(shuffled_weight_norm[self.P[idx1], :] * self.penalty[idx1, :] + shuffled_weight_norm[self.P[idx2], :] * self.penalty[idx2, :])
            loss_exchanged = torch.sum(shuffled_weight_norm[self.P[idx2], :] * self.penalty[idx1, :] + shuffled_weight_norm[self.P[idx1], :] * self.penalty[idx2, :])
        else:
            shuffled_weight_norm = self.weight_norm[self.P, :]
            loss = torch.sum(shuffled_weight_norm[:, self.Q[idx1]] * self.penalty[:, idx1] + shuffled_weight_norm[:, self.Q[idx2]] * self.penalty[:, idx2])
            loss_exchanged = torch.sum(shuffled_weight_norm[:, self.Q[idx2]] * self.penalty[:, idx1] + shuffled_weight_norm[:, self.Q[idx1]] * self.penalty[:, idx2])
        return True if loss_exchanged < loss else False
    
    @torch.no_grad()
    def stochastic_exchange(self, iters=1):
        for i in range(iters):
            idx1, idx2 = np.random.choice(self.out_channels, size=2, replace=False)
            if self.compare_loss(idx1, idx2, row=True):
                tmp = self.P[idx1].clone()
                self.P[idx1] = self.P[idx2]
                self.P[idx2] = tmp
            idx1, idx2 = np.random.choice(self.in_channels, size=2, replace=False)
            if self.compare_loss(idx1, idx2, row=False):
                tmp = self.Q[idx1].clone()
                self.Q[idx1] = self.Q[idx2]
                self.Q[idx2] = tmp

    def compute_regularity(self):
        shuffled_weight_norm = self.weight_norm[self.P, :][:, self.Q]
        return torch.sum(shuffled_weight_norm * self.penalty)
    
    @torch.no_grad()
    def mask_group(self, group_level):
        self.groupeded = True
        mask = get_mask_from_level(group_level, self.out_channels, self.in_channels)
        _, P_inv = torch.sort(self.P)
        _, Q_inv = torch.sort(self.Q)
        self.permuted_mask = mask[P_inv, :][:, Q_inv]
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

def conv1x1(in_planes, out_planes, stride=1, group=False):
    " 1x1 convolution "
    return GroupableConv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False) if group \
           else nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


def conv3x3(in_planes, out_planes, stride=1, group=True):
    " 3x3 convolution with padding "
    return GroupableConv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False) if group \
           else nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None, group1x1=False):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride, group=True)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, group=True)
        self.bn2 = nn.BatchNorm2d(planes)
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
    def __init__(self, inplanes, planes, stride=1, downsample=None, group1x1=False):
        super(Bottleneck, self).__init__()
        self.conv1 = conv1x1(inplanes, planes, group=group1x1)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, stride, group=True)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion, group=group1x1)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
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
    def __init__(self, block, layers, num_classes=1000, group1x1=False):
        super(ResNet, self).__init__()
        self.group1x1 = group1x1
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, GroupableConv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                    conv1x1(self.inplanes, planes * block.expansion, stride, group=self.group1x1),
                    nn.BatchNorm2d(planes * block.expansion),
                    )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, group1x1=self.group1x1))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, group1x1=self.group1x1))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


def resnet18(**kwargs):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    return ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)


def resnet34(**kwargs):
    """Constructs a ResNet-34 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    return ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)


def resnet50(**kwargs):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    return ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)


def resnet101(**kwargs):
    """Constructs a ResNet-101 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    return ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)


def resnet152(**kwargs):
    """Constructs a ResNet-152 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    return ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)

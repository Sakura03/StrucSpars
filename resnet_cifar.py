import torch, math
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

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

class GroupableConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, bias=True):
        super(GroupableConv2d, self).__init__(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                              stride=stride, padding=padding, dilation=dilation, bias=bias)
        self.register_buffer("P", torch.arange(self.out_channels))
        self.register_buffer("Q", torch.arange(self.in_channels))
        self.register_buffer("penalty", get_penalty_matrix(self.out_channels, self.in_channels, args.power))
        self.grouped = False
        
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
        if loss_exchanged < loss:
            return True
        else:
            return False
    
    @torch.no_grad()
    def stochastic_exchange(self, iters=1):
        for i in range(iters):
            idx1, idx2 = np.random.choice(self.out_channels, size=2, replace=False)
            if self.compare_loss(self.penalty, idx1, idx2, row=True):
                tmp = self.P[idx1].clone()
                self.P[idx1] = self.P[idx2]
                self.P[idx2] = tmp
            idx1, idx2 = np.random.choice(self.in_channels, size=2, replace=False)
            if self.compare_loss(self.penalty, idx1, idx2, row=False):
                tmp = self.Q[idx1].clone()
                self.Q[idx1] = self.Q[idx2]
                self.Q[idx2] = tmp

    def compute_regularity(self, penalty):
        shuffled_weight_norm = self.weight_norm[self.P, :][:, self.Q]
        return torch.sum(shuffled_weight_norm * penalty)
    
    @torch.no_grad()
    def real_group(self, group_level):
        self.grouped = True
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
        if self.grouped:
            x = x[:, self.Q, :, :]
        out = F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        if self.grouped:
            out = out[:, self.P_inv, :, :]
        return out

def conv1x1(in_planes, out_planes, stride=1, group=False):
    " 1x1 convolution "
    return GroupableConv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False) if group \
           else nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


def conv3x3(in_planes, out_planes, stride=1, group=True):
    " 3x3 convolution with padding "
    return GroupableConv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False) if group \
           else nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion=1
    def __init__(self, inplanes, planes, stride=1, downsample=None, group1x1=False):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion=4
    def __init__(self, inplanes, planes, stride=1, downsample=None, group1x1=False):
        super(Bottleneck, self).__init__()
        self.conv1 = conv1x1(inplanes, planes, group=group1x1)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, stride=stride, group=True)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv1x1(planes, planes*4, group=group1x1)
        self.bn3 = nn.BatchNorm2d(planes*4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=10, group1x1=False):
        super(ResNet, self).__init__()
        self.group1x1 = group1x1
        self.inplanes = 16
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 16, layers[0])
        self.layer2 = self._make_layer(block, 32, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 64, layers[2], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=1)
        self.fc = nn.Linear(64 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                    conv1x1(self.inplanes, planes*block.expansion, stride=stride, group=self.group1x1),
                    nn.BatchNorm2d(planes * block.expansion)
                    )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.group1x1))
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

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

def resnet20(**kwargs):
    model = ResNet(BasicBlock, [3, 3, 3], **kwargs)
    return model

def resnet32(**kwargs):
    model = ResNet(BasicBlock, [5, 5, 5], **kwargs)
    return model

def resnet44(**kwargs):
    model = ResNet(BasicBlock, [7, 7, 7], **kwargs)
    return model

def resnet56(**kwargs):
    model = ResNet(BasicBlock, [9, 9, 9], **kwargs)
    return model

def resnet110(**kwargs):
    model = ResNet(BasicBlock, [18, 18, 18], **kwargs)
    return model

def resnet1202(**kwargs):
    model = ResNet(BasicBlock, [200, 200, 200], **kwargs)
    return model

def resnet164(**kwargs):
    model = ResNet(Bottleneck, [18, 18, 18], **kwargs)
    return model

def resnet1001(**kwargs):
    model = ResNet(Bottleneck, [111, 111, 111], **kwargs)
    return model

if __name__ == '__main__':
    net = resnet164(group1x1=True)
    y = net(torch.randn(2, 3, 64, 64))
    print(net)
    print(y.size())


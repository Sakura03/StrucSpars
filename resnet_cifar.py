import torch
from torch import Tensor
import torch.nn as nn
from groupconv import GroupableConv2d
from typing import Any, List


class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(self, in_planes: int, planes: int, stride: int = 1, power: float = 0.5) -> None:
        super(BasicBlock, self).__init__()
        self.conv1 = GroupableConv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False, power=power)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = GroupableConv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False, power=power)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)

        self.downsample = None
        if stride != 1 or in_planes != planes:
            self.downsample = nn.Sequential(
                GroupableConv2d(in_planes, self.expansion*planes, 1, stride, bias=False, power=power),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x: Tensor) -> Tensor:
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


class ResNet(nn.Module):
    def __init__(self, block: BasicBlock, layers: List[int], num_classes: int = 10, power: float = 0.5) -> None:
        super(ResNet, self).__init__()
        self.inplanes = 16
        self.power = power
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=1, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 16, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 32, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 64, layers[2], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64 * block.expansion, num_classes)

        for m in self.modules:
            if isinstance(m, nn.Linear) or isinstance(m, (nn.Conv2d, GroupableConv2d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block: BasicBlock, planes: int, layers: int, stride: int = 1) -> nn.Sequential:
        layers = []
        layers.append(block(self.inplanes, planes, stride, power=self.power))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, norm_layer=norm_layer, power=self.power))

        return nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x


def resnet20(**kwargs: Any) -> ResNet:
    return ResNet(BasicBlock, [3, 3, 3], **kwargs)


def resnet32(**kwargs: Any) -> ResNet:
    return ResNet(BasicBlock, [5, 5, 5], **kwargs)


def resnet44(**kwargs: Any) -> ResNet:
    return ResNet(BasicBlock, [7, 7, 7], **kwargs)


def resnet56(**kwargs: Any) -> ResNet:
    return ResNet(BasicBlock, [9, 9, 9], **kwargs)


def resnet110(**kwargs: Any) -> ResNet:
    return ResNet(BasicBlock, [18, 18, 18], **kwargs)

"""ResNet

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
"""

import torch.nn as nn
import torch.nn.functional as F


def _reset_parameters(module_children):
    for module_l0 in module_children:
        if _is_empty(module_l0.children()):
            # Level 0
            if not _is_empty(module_l0.parameters()):
                module_l0.reset_parameters()
        else:
            # Level 1
            for module_l1 in module_l0.children():
                if not _is_empty(module_l1.parameters()):
                    module_l1.reset_parameters()


def _is_empty(iterable):
    try:
        next(iterable)
    except StopIteration:
        return True
    return False


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

    def reset_parameters(self):
        _reset_parameters(self.children())


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

    def reset_parameters(self):
        _reset_parameters(self.children())


class ResNetFront(nn.Module):
    def __init__(
        self, block=BasicBlock, num_blocks=None, num_channels=1, num_features=1, num_parts=1,
    ):
        super().__init__()
        if num_blocks is None:
            num_blocks = [2, 2, 2, 2]

        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512 * block.expansion, 10)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


class ResNetCenter(nn.Module):
    def __init__(self, block=BasicBlock, num_channels=1, num_features=1, num_parts=1):
        super().__init__()
        self.linear = nn.Linear(512 * block.expansion, 512)

    def forward(self, x):
        out = self.linear(x)
        return out


class ResNetBack(nn.Module):
    def __init__(
        self, block=BasicBlock, num_classes=10, num_channels=1, num_features=1, num_parts=1
    ):
        super().__init__()
        self.linear = nn.Linear(512 * block.expansion, num_classes)

    def forward(self, x):
        # out = self.linear(x)
        # out = F.log_softmax(x, dim=1)
        # return out
        return x


def ResNet18Front(num_channels=1, num_features=1, num_parts=1, load=False, freeze=False):
    return ResNetFront(
        BasicBlock,
        [2, 2, 2, 2],
        num_channels=num_channels,
        num_features=num_features,
        num_parts=num_parts,
        load=load,
        freeze=freeze,
    )


def ResNet18Center(num_channels=1, num_features=1, num_parts=1, load=False, freeze=False):
    return ResNetCenter(
        BasicBlock, num_channels=num_channels, num_features=num_features, num_parts=num_parts, load=load, freeze=freeze
    )


def ResNet18Back(num_channels=1, num_features=1, num_parts=1, load=False, freeze=False):
    return ResNetBack(
        BasicBlock,
        num_classes=10,
        num_channels=num_channels,
        num_features=num_features,
        num_parts=num_parts,
        load=load,
        freeze=freeze,
    )


# def ResNet18():
#     return ResNet(BasicBlock, [2, 2, 2, 2])
#
#
# def ResNet34():
#     return ResNet(BasicBlock, [3, 4, 6, 3])
#
#
# def ResNet50():
#     return ResNet(Bottleneck, [3, 4, 6, 3])
#
#
# def ResNet101():
#     return ResNet(Bottleneck, [3, 4, 23, 3])
#
#
# def ResNet152():
#     return ResNet(Bottleneck, [3, 8, 36, 3])


# def test():
#     net = ResNet18()
#     y = net(torch.randn(1, 3, 32, 32))
#     print(y.size())

# test()

import torch
import torch.nn as nn


cfg = {
    "VGG11": [64, "M", 128, "M", "C", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "VGG13": [64, 64, "M", 128, 128, "M", "C", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "VGG16": [64, 64, "M", 128, 128, "M", "C", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "M"],
    "VGG19": [
        64,
        64,
        "M",
        128,
        128,
        "M",
        "C",
        256,
        256,
        256,
        256,
        "M",
        512,
        512,
        512,
        512,
        "M",
        512,
        512,
        512,
        512,
        "M",
    ],
}


# class VGG(nn.Module):
#     def __init__(self, vgg_name):
#         super().__init__()
#         self.features = self._make_layers(cfg[vgg_name])
#         self.classifier = nn.Linear(512, 10)
#
#     def forward(self, x):
#         out = self.features(x)
#         out = out.view(out.size(0), -1)
#         out = self.classifier(out)
#         return out
#
#     def _make_layers(self, cfg):
#         layers = []
#         in_channels = 3
#         for x in cfg:
#             if x == 'M':
#                 layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
#             else:
#                 layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
#                            nn.BatchNorm2d(x),
#                            nn.ReLU(inplace=True)]
#                 in_channels = x
#         layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
#         return nn.Sequential(*layers)


class VGGFront(nn.Module):
    def __init__(
        self, vgg_name, num_channels=1, num_features=1, num_parts=1, with_center=False, load=False, freeze=False
    ):
        super().__init__()
        self.with_center = with_center
        self.load = load
        self.freeze = freeze
        self.features = self._make_layers(cfg[vgg_name], with_center)

    def forward(self, x):
        out = self.features(x)
        if not self.with_center:
            out = out.view(out.size(0), -1)
        return out

    def _make_layers(self, cfg, with_center):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == "C":
                if with_center:
                    break
            elif x == "M":
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [
                    nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                    nn.BatchNorm2d(x),
                    nn.ReLU(inplace=True),
                ]
                in_channels = x
        if not with_center:
            layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)


class VGGCenter(nn.Module):
    def __init__(
        self, vgg_name, num_channels=1, num_features=1, num_parts=1, with_center=False, load=False, freeze=False
    ):
        super().__init__()
        self.load = load
        self.freeze = freeze
        self.features = self._make_layers(cfg[vgg_name])

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 64
        enable = False
        for x in cfg:
            if x == "C":
                enable = True
            elif x == "M":
                if enable:
                    layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                if enable:
                    layers += [
                        nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                        nn.BatchNorm2d(x),
                        nn.ReLU(inplace=True),
                    ]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)


class VGGBack(nn.Module):
    def __init__(
        self,
        vgg_name,
        num_classes,
        num_channels=1,
        num_features=1,
        num_parts=1,
        with_center=False,
        load=False,
        freeze=False,
    ):
        super().__init__()
        self.load = load
        self.freeze = freeze
        self.classifier = nn.Linear(512, num_classes)

    def forward(self, x):
        out = self.classifier(x)
        return out


def VGG11Front(num_channels=1, num_features=1, num_parts=1, with_center=False, load=False, freeze=False):
    return VGGFront(
        "VGG11",
        num_channels=num_channels,
        num_features=num_features,
        num_parts=num_parts,
        with_center=with_center,
        load=load,
        freeze=freeze,
    )


def VGG11Center(num_channels=1, num_features=1, num_parts=1, load=False, freeze=False):
    return VGGCenter(
        "VGG11",
        num_channels=num_channels,
        num_features=num_features,
        num_parts=num_parts,
        load=load,
        freeze=freeze,
    )


def VGG11Back(num_channels=1, num_features=1, num_parts=1, load=False, freeze=False):
    return VGGBack(
        "VGG11",
        num_classes=10,
        num_channels=num_channels,
        num_features=num_features,
        num_parts=num_parts,
        load=load,
        freeze=freeze,
    )


def VGG13Front(num_channels=1, num_features=1, num_parts=1, with_center=False, load=False, freeze=False):
    return VGGFront(
        "VGG13",
        num_channels=num_channels,
        num_features=num_features,
        num_parts=num_parts,
        with_center=with_center,
        load=load,
        freeze=freeze,
    )


def VGG13Center(num_channels=1, num_features=1, num_parts=1, load=False, freeze=False):
    return VGGCenter(
        "VGG13",
        num_channels=num_channels,
        num_features=num_features,
        num_parts=num_parts,
        load=load,
        freeze=freeze,
    )


def VGG13Back(num_channels=1, num_features=1, num_parts=1, load=False, freeze=False):
    return VGGBack(
        "VGG13",
        num_classes=10,
        num_channels=num_channels,
        num_features=num_features,
        num_parts=num_parts,
        load=load,
        freeze=freeze,
    )


def VGG16Front(num_channels=1, num_features=1, num_parts=1, with_center=False, load=False, freeze=False):
    return VGGFront(
        "VGG16",
        num_channels=num_channels,
        num_features=num_features,
        num_parts=num_parts,
        with_center=with_center,
        load=load,
        freeze=freeze,
    )


def VGG16Center(num_channels=1, num_features=1, num_parts=1, load=False, freeze=False):
    return VGGCenter(
        "VGG16",
        num_channels=num_channels,
        num_features=num_features,
        num_parts=num_parts,
        load=load,
        freeze=freeze,
    )


def VGG16Back(num_channels=1, num_features=1, num_parts=1, load=False, freeze=False):
    return VGGBack(
        "VGG16",
        num_classes=10,
        num_channels=num_channels,
        num_features=num_features,
        num_parts=num_parts,
        load=load,
        freeze=freeze,
    )


def VGG19Front(num_channels=1, num_features=1, num_parts=1, with_center=False, load=False, freeze=False):
    return VGGFront(
        "VGG19",
        num_channels=num_channels,
        num_features=num_features,
        num_parts=num_parts,
        with_center=with_center,
        load=load,
        freeze=freeze,
    )


def VGG19Center(num_channels=1, num_features=1, num_parts=1, load=False, freeze=False):
    return VGGCenter(
        "VGG19",
        num_channels=num_channels,
        num_features=num_features,
        num_parts=num_parts,
        load=load,
        freeze=freeze,
    )


def VGG19Back(num_channels=1, num_features=1, num_parts=1, load=False, freeze=False):
    return VGGBack(
        "VGG19",
        num_classes=10,
        num_channels=num_channels,
        num_features=num_features,
        num_parts=num_parts,
        load=load,
        freeze=freeze,
    )


def test_front():
    net = VGG11Front(num_channels=1, num_features=1, num_parts=1, with_center=False, load=False, freeze=False)
    x = torch.randn(2, 3, 32, 32)
    y = net(x)
    print(y.size())


def test_center():
    net = VGG11Center(num_channels=1, num_features=1, num_parts=1, load=False, freeze=False)
    x = torch.randn(2, 3, 8, 8)
    y = net(x)
    print(y.size())


def test_back():
    net = VGG11Back(num_channels=1, num_features=1, num_parts=1, load=False, freeze=False)
    x = torch.randn(2, 512)
    y = net(x)
    print(y.size())

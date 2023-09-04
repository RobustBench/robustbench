# Code adapted from https://github.com/wzekai99/DM-Improves-AT
from typing import Tuple, Union, Optional, Callable
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops.misc import SqueezeExcitation

from robustbench.model_zoo.architectures.dm_wide_resnet import CIFAR10_MEAN, CIFAR10_STD


class _Block(nn.Module):
    def __init__(
        self,
        in_planes,
        out_planes,
        stride,
        groups,
        activation_fn=nn.ReLU,
        se_ratio=None,
        se_activation=nn.ReLU,
        se_order=1,
    ):
        super().__init__()
        self.batchnorm_0 = nn.BatchNorm2d(in_planes, momentum=0.01)
        self.relu_0 = activation_fn(inplace=True)
        self.conv_0 = nn.Conv2d(
            in_planes,
            out_planes,
            kernel_size=3,
            stride=stride,
            groups=groups,
            padding=0,
            bias=False,
        )
        self.batchnorm_1 = nn.BatchNorm2d(out_planes, momentum=0.01)
        self.relu_1 = activation_fn(inplace=True)
        self.conv_1 = nn.Conv2d(
            out_planes,
            out_planes,
            kernel_size=3,
            stride=1,
            groups=groups,
            padding=1,
            bias=False,
        )
        self.has_shortcut = in_planes != out_planes
        if self.has_shortcut:
            self.shortcut = nn.Conv2d(
                in_planes,
                out_planes,
                kernel_size=1,
                stride=stride,
                padding=0,
                bias=False,
            )
        else:
            self.shortcut = None
        self._stride = stride

        self.se = None
        if se_ratio:
            assert se_activation is not None
            width_se_out = int(round(se_ratio * out_planes))
            self.se = SqueezeExcitation(
                input_channels=out_planes,
                squeeze_channels=width_se_out,
                activation=se_activation,
            )
            self.se_order = se_order

    def forward(self, x):
        if self.has_shortcut:
            x = self.relu_0(self.batchnorm_0(x))
        else:
            out = self.relu_0(self.batchnorm_0(x))
        v = x if self.has_shortcut else out
        if self._stride == 1:
            v = F.pad(v, (1, 1, 1, 1))
        elif self._stride == 2:
            v = F.pad(v, (0, 1, 0, 1))
        else:
            raise ValueError("Unsupported `stride`.")
        out = self.conv_0(v)

        if self.se and self.se_order == 1:
            out = self.se(out)

        out = self.relu_1(self.batchnorm_1(out))
        out = self.conv_1(out)

        if self.se and self.se_order == 2:
            out = self.se(out)

        out = torch.add(self.shortcut(x) if self.has_shortcut else x, out)
        return out


class _BlockGroup(nn.Module):
    def __init__(
        self,
        num_blocks,
        in_planes,
        out_planes,
        stride,
        groups,
        activation_fn=nn.ReLU,
        se_ratio=None,
        se_activation=nn.ReLU,
        se_order=1,
    ):
        super().__init__()
        block = []
        for i in range(num_blocks):
            block.append(
                _Block(
                    i == 0 and in_planes or out_planes,
                    out_planes,
                    i == 0 and stride or 1,
                    groups=groups,
                    activation_fn=activation_fn,
                    se_ratio=se_ratio,
                    se_activation=se_activation,
                    se_order=se_order,
                )
            )
        self.block = nn.Sequential(*block)

    def forward(self, x):
        return self.block(x)


class NormalizedWideResNet(nn.Module):
    def __init__(
        self,
        mean: Tuple[float],
        std: Tuple[float],
        stem_width: int,
        depth: Tuple[int],
        stage_width: Tuple[int],
        groups: Tuple[int],
        activation_fn: nn.Module,
        se_ratio: Optional[float],
        se_activation: Optional[Callable[..., nn.Module]],
        se_order: Optional[int],
        num_classes: int = 10,
        padding: int = 0,
        num_input_channels: int = 3,
    ):
        super().__init__()
        self.mean = torch.tensor(mean).view(num_input_channels, 1, 1)
        self.std = torch.tensor(std).view(num_input_channels, 1, 1)
        self.mean_cuda = None
        self.std_cuda = None
        self.padding = padding
        num_channels = [stem_width, *stage_width]
        self.init_conv = nn.Conv2d(
            num_input_channels,
            num_channels[0],
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.layer = nn.Sequential(
            _BlockGroup(
                depth[0],
                num_channels[0],
                num_channels[1],
                1,
                groups=groups[0],
                activation_fn=activation_fn,
                se_ratio=se_ratio,
                se_activation=se_activation,
                se_order=se_order,
            ),
            _BlockGroup(
                depth[1],
                num_channels[1],
                num_channels[2],
                2,
                groups=groups[1],
                activation_fn=activation_fn,
                se_ratio=se_ratio,
                se_activation=se_activation,
                se_order=se_order,
            ),
            _BlockGroup(
                depth[2],
                num_channels[2],
                num_channels[3],
                2,
                groups=groups[2],
                activation_fn=activation_fn,
                se_ratio=se_ratio,
                se_activation=se_activation,
                se_order=se_order,
            ),
        )
        self.batchnorm = nn.BatchNorm2d(num_channels[3], momentum=0.01)
        self.relu = activation_fn(inplace=True)
        self.logits = nn.Linear(num_channels[3], num_classes)
        self.num_channels = num_channels[3]

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, x):
        if self.padding > 0:
            x = F.pad(x, (self.padding,) * 4)
        if x.is_cuda:
            if self.mean_cuda is None:
                self.mean_cuda = self.mean.cuda()
                self.std_cuda = self.std.cuda()
            out = (x - self.mean_cuda) / self.std_cuda
        else:
            out = (x - self.mean) / self.std

        out = self.init_conv(out)
        out = self.layer(out)
        out = self.relu(self.batchnorm(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(-1, self.num_channels)
        return self.logits(out)


def get_model(model_name):

    if model_name == 'ra_wrn70_16':
        model = NormalizedWideResNet(
            CIFAR10_MEAN,
            CIFAR10_STD,
            stem_width=96,
            depth=[30, 31, 10],
            stage_width=[216, 432, 864],
            groups=[1, 1, 1],
            activation_fn=torch.nn.SiLU,
            se_ratio=0.25,
            se_activation=torch.nn.ReLU,
            se_order=2, num_classes=10,
            )
    else:
        raise ValueError(f"Unknown model name: {model_name}.")

    return model


if __name__ == '__main__':

    model = get_model('ra_wrn70_16')
    model.cuda()
    x = torch.rand([10, 3, 32, 32])
    with torch.no_grad():
        print(model(x.cuda()).shape)


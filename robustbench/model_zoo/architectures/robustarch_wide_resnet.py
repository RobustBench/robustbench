# Code adapted from https://github.com/wzekai99/DM-Improves-AT
from typing import (
    Tuple,
    Optional,
    Callable,
    Any,
    Callable,
    Optional,
    Tuple,
    # List,  # TODO: for python<3.9 one needs to use `List[]` instead of `list[]`.
)
import math
import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torchvision.ops.misc import SqueezeExcitation
from collections import OrderedDict
from torchvision.ops.misc import Conv2dNormActivation, SqueezeExcitation

from robustbench.model_zoo.architectures.dm_wide_resnet import CIFAR10_MEAN, CIFAR10_STD


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

INPLACE_ACTIVATIONS = [nn.ReLU]
NORMALIZATIONS = [nn.BatchNorm2d, nn.InstanceNorm2d, nn.GroupNorm, nn.LayerNorm]


def normalize_fn(tensor, mean, std):
    mean = mean[None, :, None, None]
    std = std[None, :, None, None]
    return tensor.sub(mean).div(std)


class NormalizeByChannelMeanStd(nn.Module):
    def __init__(self, mean, std):
        super(NormalizeByChannelMeanStd, self).__init__()
        if not isinstance(mean, torch.Tensor):
            mean = torch.tensor(mean)
        if not isinstance(std, torch.Tensor):
            std = torch.tensor(std)
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)

    def forward(self, tensor):
        return normalize_fn(tensor, self.mean, self.std)

    def extra_repr(self):
        return "mean={}, std={}".format(self.mean, self.std)


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
            #if self.mean_cuda is None
            self.mean_cuda = self.mean.to(x.device)  # TODO: improve this.
            self.std_cuda = self.std.to(x.device)
            out = (x - self.mean_cuda) / self.std_cuda
        else:
            out = (x - self.mean) / self.std

        out = self.init_conv(out)
        out = self.layer(out)
        out = self.relu(self.batchnorm(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(-1, self.num_channels)
        return self.logits(out)


class NormActivationConv(torch.nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: Optional[int] = None,
        groups: int = 1,
        norm_layer: Optional[Callable[..., torch.nn.Module]] = torch.nn.BatchNorm2d,
        activation_layer: Optional[Callable[..., torch.nn.Module]] = torch.nn.ReLU,
        dilation: int = 1,
        inplace: Optional[bool] = True,
        bias: Optional[bool] = None,
        conv_layer: Callable[..., torch.nn.Module] = torch.nn.Conv2d,
    ) -> None:
        if padding is None:
            padding = (kernel_size - 1) // 2 * dilation
        if bias is None:
            bias = norm_layer is None

        layers = list()

        if norm_layer is not None:
            layers.append(norm_layer(in_channels))

        if activation_layer is not None:
            params = {} if inplace is None else {"inplace": inplace}
            layers.append(activation_layer(**params))

        layers.append(
            conv_layer(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                dilation=dilation,
                groups=groups,
                bias=bias,
            )
        )
        super().__init__(*layers)
        self.out_channels = out_channels


class NormActivationConv2d(NormActivationConv):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: Optional[int] = None,
        groups: int = 1,
        norm_layer: Optional[Callable[..., torch.nn.Module]] = torch.nn.BatchNorm2d,
        activation_layer: Optional[Callable[..., torch.nn.Module]] = torch.nn.ReLU,
        dilation: int = 1,
        inplace: Optional[bool] = True,
        bias: Optional[bool] = None,
    ) -> None:
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            groups,
            norm_layer,
            activation_layer,
            dilation,
            inplace,
            bias,
            torch.nn.Conv2d,
        )


class BottleneckTransform(nn.Sequential):
    "Transformation in a Bottleneck: 1x1, kxk (k=3, 5, 7, ...) [+SE], 1x1"
    "Supported archs: [preact] [norm func+num] [act func+num] [conv kernel]"

    def __init__(
        self,
        width_in: int,
        width_out: int,
        kernel: int,
        stride: int,
        dilation: int,
        norm_layer: list[Callable[..., nn.Module]],
        activation_layer: list[Callable[..., nn.Module]],
        group_width: int,
        bottleneck_multiplier: float,
        se_ratio: Optional[float],
        se_activation: Optional[Callable[..., nn.Module]],
        ConvBlock: Callable[..., nn.Module],
    ):
        # compute transform params
        w_b = int(
            round(width_out * bottleneck_multiplier)
        )  # bottleneck_multiplier > 1 for inverted bottleneck
        g = w_b // group_width
        assert len(norm_layer) == 3
        assert len(activation_layer) == 3
        assert g > 0, f"Group convolution groups {g} should be greater than 0."
        assert (
            w_b % g == 0
        ), f"Convolution input channels {w_b} is not divisible by {g} groups."

        layers: OrderedDict[str, nn.Module] = OrderedDict()
        layers["a"] = ConvBlock(
            width_in,
            w_b,
            kernel_size=1,
            stride=1,
            norm_layer=norm_layer[0],
            activation_layer=activation_layer[0],
            inplace=True if activation_layer[0] in INPLACE_ACTIVATIONS else None,
        )

        layers["b"] = ConvBlock(
            w_b,
            w_b,
            kernel,
            stride=stride,
            groups=g,
            dilation=dilation,
            norm_layer=norm_layer[1],
            activation_layer=activation_layer[1],
            inplace=True if activation_layer[1] in INPLACE_ACTIVATIONS else None,
        )

        if se_ratio:
            assert se_activation is not None
            width_se_out = int(round(se_ratio * width_in))
            layers["se"] = SqueezeExcitation(
                input_channels=w_b,
                squeeze_channels=width_se_out,
                activation=se_activation,
            )
        if ConvBlock == Conv2dNormActivation:
            layers["c"] = ConvBlock(
                w_b,
                width_out,
                kernel_size=1,
                stride=1,
                norm_layer=norm_layer[2],
                activation_layer=None,
            )
        else:
            layers["c"] = ConvBlock(
                w_b,
                width_out,
                kernel_size=1,
                stride=1,
                norm_layer=norm_layer[2],
                activation_layer=activation_layer[2],
                inplace=True if activation_layer[2] in INPLACE_ACTIVATIONS else None,
            )

        super().__init__(layers)


class BottleneckBlock(nn.Module):
    """Bottleneck block x + F(x), where F = bottleneck transform"""

    def __init__(
        self,
        width_in: int,
        width_out: int,
        kernel: int,
        stride: int,
        dilation: int,
        norm_layer: list[Callable[..., nn.Module]],
        activation_layer: list[Callable[..., nn.Module]],
        group_width: int,
        bottleneck_multiplier: float,
        se_ratio: Optional[float],
        se_activation: Optional[Callable[..., nn.Module]],
        ConvBlock: Callable[..., nn.Module],
        downsample_norm: Callable[..., nn.Module],
    ) -> None:
        super().__init__()

        # projection on skip connection if shape changes
        self.proj = None
        should_proj = (width_in != width_out) or (stride != 1)
        if should_proj:
            if ConvBlock == Conv2dNormActivation:
                self.proj = ConvBlock(
                    width_in,
                    width_out,
                    kernel_size=1,
                    stride=stride,
                    norm_layer=downsample_norm,
                    activation_layer=None,
                )
            elif ConvBlock == NormActivationConv2d:
                self.proj = ConvBlock(
                    width_in,
                    width_out,
                    kernel_size=1,
                    stride=stride,
                    norm_layer=None,
                    activation_layer=None,
                    bias=False,
                )

        self.F = BottleneckTransform(
            width_in,
            width_out,
            kernel,
            stride,
            dilation,
            norm_layer,
            activation_layer,
            group_width,
            bottleneck_multiplier,
            se_ratio,
            se_activation,
            ConvBlock,
        )

        if ConvBlock == Conv2dNormActivation:
            if activation_layer[2] is not None:
                if activation_layer[2] in INPLACE_ACTIVATIONS:
                    self.last_activation = activation_layer[2](inplace=True)
                else:
                    self.last_activation = activation_layer[2]()
        else:
            self.last_activation = None

    def forward(self, x: Tensor) -> Tensor:
        if self.proj is not None:
            x = self.proj(x) + self.F(x)
        else:
            x = x + self.F(x)

        if self.last_activation is not None:
            return self.last_activation(x)
        else:
            return x


class Stage(nn.Sequential):
    """Stage is a sequence of blocks with the same output shape. Downsampling block is the first in each stage"""

    """Options: stage numbers, stage depth, dense connection"""

    def __init__(
        self,
        width_in: int,
        width_out: int,
        kernel: int,
        stride: int,
        dilation: int,
        norm_layer: list[Callable[..., nn.Module]],
        activation_layer: list[Callable[..., nn.Module]],
        group_width: int,
        bottleneck_multiplier: float,
        se_ratio: Optional[float],
        se_activation: Optional[Callable[..., nn.Module]],
        ConvBlock: Callable[..., nn.Module],
        downsample_norm: Callable[..., nn.Module],
        depth: int,
        dense_ratio: Optional[float],
        block_constructor: Callable[..., nn.Module] = BottleneckBlock,
        stage_index: int = 0,
    ):
        super().__init__()
        self.dense_ratio = dense_ratio
        for i in range(depth):
            block = block_constructor(
                width_in if i == 0 else width_out,
                width_out,
                kernel,
                stride if i == 0 else 1,
                dilation,
                norm_layer,
                activation_layer,
                group_width,
                bottleneck_multiplier,
                se_ratio,
                se_activation,
                ConvBlock,
                downsample_norm,
            )

            self.add_module(f"stage{stage_index}-block{i}", block)

    def forward(self, x: Tensor) -> Tensor:
        if self.dense_ratio:
            assert self.dense_ratio > 0
            features = list([x])
            for i, module in enumerate(self):
                input = features[-1]
                if i > 2:
                    for j in range(self.dense_ratio):
                        if j + 4 > len(features):
                            break
                        input = input + features[-3 - j]
                x = module(input)
                features.append(x)

            # output of each stage is also densely connected
            x = features[-1]
            for k in range(self.dense_ratio):
                if k + 4 > len(features):
                    break
                x = x + features[-3 - k]
        else:
            for module in self:
                x = module(x)
        return x


class Stem(nn.Module):
    """Stem for ImageNet: kxk, BN, ReLU[, MaxPool]"""

    def __init__(
        self,
        width_in: int,
        width_out: int,
        kernel_size: int,
        norm_layer: Callable[..., nn.Module],
        activation_layer: Callable[..., nn.Module],
        downsample_factor: int,
        patch_size: Optional[int],
    ) -> None:
        super().__init__()

        assert downsample_factor % 2 == 0 and downsample_factor >= 2
        layers: OrderedDict[str, nn.Module] = OrderedDict()

        stride = 2
        if patch_size:
            kernel_size = patch_size
            stride = patch_size

        layers["stem"] = Conv2dNormActivation(
            width_in,
            width_out,
            kernel_size=kernel_size,
            stride=stride,
            norm_layer=norm_layer,
            activation_layer=activation_layer,
        )

        if not patch_size and downsample_factor // 2 > 1:
            layers["stem_downsample"] = nn.MaxPool2d(
                kernel_size=3, stride=downsample_factor // 2, padding=1
            )

        self.stem = nn.Sequential(layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.stem(x)


class ConfigurableModel(nn.Module):
    def __init__(
        self,
        stage_widths: list[int],  # output width of each stage
        kernel: int,  # kernel for non-pointwise conv
        strides: list[int],  # stride in each stage
        dilation: int,  # dilation for non-pointwise conv
        norm_layer: list[
            Callable[..., nn.Module]
        ],  # norm layer in each block, length 3 for bottleneck
        activation_layer: list[
            Callable[..., nn.Module]
        ],  # activation layer in each block, length 3 for bottleneck
        group_widths: list[
            int
        ],  # group conv width in each stage, groups = width_out * bottleneck_multiplier // group_width
        bottleneck_multipliers: list[
            float
        ],  # bottleneck_multiplier > 1 for inverted bottleneck
        downsample_norm: Callable[
            ..., nn.Module
        ],  # norm layer in downsampling shortcut
        depths: list[int],  # depth in each stage
        dense_ratio: Optional[float],  # dense connection ratio
        stem_type: Callable[..., nn.Module],  # stem stage
        stem_width: int,  # stem stage output width
        stem_kernel: int,  # stem stage kernel size
        stem_downsample_factor: int,  # downscale factor in the stem stage, if > 2, a maxpool layer is added
        stem_patch_size: Optional[int],  # patchify stem patch size
        block_constructor: Callable[
            ..., nn.Module
        ] = BottleneckBlock,  # block type in body stage
        ConvBlock: Callable[
            ..., nn.Module
        ] = Conv2dNormActivation,  # block with different "conv-norm-act" order
        se_ratio: Optional[float] = None,  # squeeze and excitation (SE) ratio
        se_activation: Optional[
            Callable[..., nn.Module]
        ] = None,  # activation layer in SE block
        weight_init_type: str = "resnet",  # initialization type
        num_classes: int = 1000,  # num of classification classes
    ) -> None:
        super().__init__()

        num_stages = len(stage_widths)
        assert len(strides) == num_stages
        assert len(bottleneck_multipliers) == num_stages
        assert len(group_widths) == num_stages
        assert len(norm_layer) == len(activation_layer)
        assert (
            sum([i % 8 for i in stage_widths]) == 0
        ), f"Stage width {stage_widths} non-divisible by 8"

        # stem
        self.stem = stem_type(
            width_in=3,
            width_out=stem_width,
            kernel_size=stem_kernel,
            norm_layer=nn.BatchNorm2d,
            activation_layer=nn.ReLU,
            downsample_factor=stem_downsample_factor,
            patch_size=stem_patch_size,
        )

        # stages
        current_width = stem_width
        stages = list()
        for i, (
            width_out,
            stride,
            group_width,
            bottleneck_multiplier,
            depth,
        ) in enumerate(
            zip(stage_widths, strides, group_widths, bottleneck_multipliers, depths)
        ):
            stages.append(
                (
                    f"stage{i + 1}",
                    Stage(
                        current_width,
                        width_out,
                        kernel,
                        stride,
                        dilation,
                        norm_layer,
                        activation_layer,
                        group_width,
                        bottleneck_multiplier,
                        se_ratio,
                        se_activation,
                        ConvBlock,
                        downsample_norm,
                        depth,
                        dense_ratio,
                        block_constructor,
                        stage_index=i + 1,
                    ),
                )
            )

            current_width = width_out

        self.stages = nn.Sequential(OrderedDict(stages))

        # classification head
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(current_width, num_classes)

        # initialization
        if weight_init_type == "resnet":
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                    nn.init.normal_(m.weight, mean=0.0, std=math.sqrt(2.0 / fan_out))
                    # nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                elif m in NORMALIZATIONS:
                    nn.init.ones_(m.weight)
                    nn.init.zeros_(m.bias)
                elif isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, mean=0.0, std=0.01)
                    nn.init.zeros_(m.bias)
        else:
            raise NotImplementedError

    def forward(self, x: Tensor) -> Tensor:
        x = self.stem(x)
        x = self.stages(x)

        x = self.avgpool(x)
        x = x.flatten(start_dim=1)
        x = self.fc(x)

        return x


class NormalizedConfigurableModel(ConfigurableModel):
    def __init__(self, mean: list[float], std: list[float], **kwargs: Any):
        super().__init__(**kwargs)

        assert len(mean) == len(std)
        self.normalization = NormalizeByChannelMeanStd(mean=mean, std=std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.normalization(x)

        x = self.stem(x)
        x = self.stages(x)

        x = self.avgpool(x)
        x = x.flatten(start_dim=1)
        x = self.fc(x)

        return x


def get_model(model_name):
    if model_name == "ra_wrn70_16":
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
            se_order=2,
            num_classes=10,
        )
    elif model_name == "ra_wrn101_2":
        model = NormalizedConfigurableModel(
            mean=IMAGENET_MEAN,
            std=IMAGENET_STD,
            stage_widths=[512, 1024, 2016, 4032],
            kernel=3,
            strides=[2, 2, 2, 2],
            dilation=1,
            norm_layer=[nn.Identity, nn.BatchNorm2d, nn.BatchNorm2d],
            activation_layer=[nn.SiLU] * 3,
            group_widths=[64, 128, 252, 504],
            bottleneck_multipliers=[0.25] * 4,
            downsample_norm=nn.BatchNorm2d,
            depths=[7, 11, 18, 1],
            dense_ratio=None,
            stem_type=Stem,
            stem_width=96,
            stem_kernel=7,
            stem_downsample_factor=2,
            stem_patch_size=None,
            block_constructor=BottleneckBlock,
            ConvBlock=Conv2dNormActivation,
            se_ratio=0.25,
            se_activation=nn.ReLU,
        )
    else:
        raise ValueError(f"Unknown model name: {model_name}.")

    return model


if __name__ == "__main__":
    model = get_model("ra_wrn70_16")
    model.cuda()
    x = torch.rand([10, 3, 32, 32])
    with torch.no_grad():
        print(model(x.cuda()).shape)

    model = get_model("ra_wrn101_2")
    #print(model.state_dict().keys())
    model.cuda()
    x = torch.rand([10, 3, 224, 224])
    with torch.no_grad():
        print(model(x.cuda()).shape)

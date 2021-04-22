import abc
from typing import Sequence

import torch
from torch import nn

from robustbench.model_zoo.architectures.resnet import PreActResNet, ResNet
from robustbench.model_zoo.architectures.resnext import CifarResNeXt
from robustbench.model_zoo.architectures.wide_resnet import WideResNet


class LipschitzModel(abc.ABC):
    def get_lipschitz_layers(self) -> Sequence[nn.Module]:
        raise NotImplementedError


class View(nn.Module):
    """https://discuss.pytorch.org/t/how-to-build-a-view-layer-in-pytorch-for-sequential-models/53958/11"""

    def __init__(self, shape):
        super().__init__()
        self.shape = shape

    def __repr__(self):
        return f'View{self.shape}'

    def forward(self, input):
        out = input.view(self.shape)
        return out


class NormalizedWideResNet(WideResNet):
    def get_lipschitz_layers(self) -> Sequence[nn.Module]:
        layers = list(super().get_lipschitz_layers())
        layers[0] = nn.Sequential(NormalizeData(self.mu, self.sigma), layers[0])
        return layers


class NormalizedResNet(ResNet):
    def get_lipschitz_layers(self) -> Sequence[nn.Module]:
        layers = list(super().get_lipschitz_layers())
        layers[0] = nn.Sequential(NormalizeData(self.mu, self.sigma), layers[0])
        return layers


class NormalizedPreActResNet(PreActResNet):
    def get_lipschitz_layers(self) -> Sequence[nn.Module]:
        layers = list(super().get_lipschitz_layers())
        layers[0] = nn.Sequential(NormalizeData(self.mu, self.sigma), layers[0])
        return layers


class NormalizedCifarResNeXt(CifarResNeXt):
    def get_lipschitz_layers(self) -> Sequence[nn.Module]:
        layers = list(super().get_lipschitz_layers())
        layers[0] = nn.Sequential(NormalizeData(self.mu, self.sigma), layers[0])
        return layers


class NormalizeData(nn.Module):
    """Normalizes data given mean and std deviation."""
    def __init__(self, mean: torch.Tensor, std: torch.Tensor):
        super().__init__()
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / self.std

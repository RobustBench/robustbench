import abc
from typing import Sequence

from torch import nn


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

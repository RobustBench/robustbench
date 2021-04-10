from typing import Sequence

import torch
from torch import nn

from robustbench.model_zoo.architectures.utils import Layer, LipschitzModel


class DummyModel(nn.Module, LipschitzModel):
    def __init__(self, in_shape=3072, out_shape=10, slope=None):
        super().__init__()
        self.in_shape = in_shape
        self.out_shape = out_shape
        self.slope = slope
        self.main = nn.Sequential(nn.Flatten(), nn.Linear(in_shape, out_shape, bias=False))

        if slope is not None:
            for parameter in self.parameters():
                parameter.data = slope * torch.ones(parameter.data.shape)

    def forward(self, x):
        return self.main(x)

    def get_lipschitz_layers(self) -> Sequence[Layer]:
        return [self.main]


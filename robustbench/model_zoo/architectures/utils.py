import abc
from typing import Callable, Sequence

import torch

Layer = Callable[[torch.Tensor], torch.Tensor]


class LipschitzModel(abc.ABC):
    def get_lipschitz_layers(self) -> Sequence[Layer]:
        raise NotImplementedError

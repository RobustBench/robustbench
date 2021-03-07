from collections import OrderedDict
from robustbench.model_zoo.architectures.resnet import PreActResNet, PreActBlock

import torch

from robustbench.model_zoo.architectures.dm_wide_resnet import CIFAR100_MEAN, CIFAR100_STD, \
    DMWideResNet, Swish
from robustbench.model_zoo.architectures.wide_resnet import WideResNet
from robustbench.model_zoo.enums import ThreatModel


class Gowal2020UncoveringNet(DMWideResNet):
    def __init__(self, depth=70, width=16):
        super().__init__(num_classes=100, depth=depth, width=width, activation_fn=Swish,
                         mean=CIFAR100_MEAN, std=CIFAR100_STD)


class Chen2020EfficientNet(WideResNet):
    def __init__(self, depth=34, widen_factor=10):
        super().__init__(depth=depth, widen_factor=widen_factor, sub_block1=False, num_classes=100)
        self.register_buffer('mu', torch.tensor(
            [0.5071, 0.4867, 0.4408]).view(1, 3, 1, 1))
        self.register_buffer('sigma', torch.tensor(
            [0.2675, 0.2565, 0.2761]).view(1, 3, 1, 1))

    def forward(self, x):
        x = (x - self.mu) / self.sigma
        return super().forward(x)


class Wu2020AdversarialNet(WideResNet):
    def __init__(self, depth=34, widen_factor=10):
        super().__init__(depth=depth, widen_factor=widen_factor, sub_block1=False, num_classes=100)
        self.register_buffer('mu', torch.tensor(
            [0.5070751592371323, 0.48654887331495095, 0.4409178433670343]).view(1, 3, 1, 1))
        self.register_buffer('sigma', torch.tensor(
            [0.2673342858792401, 0.2564384629170883, 0.27615047132568404]).view(1, 3, 1, 1))

    def forward(self, x):
        x = (x - self.mu) / self.sigma
        return super().forward(x)


class Rice2020OverfittingNet(PreActResNet):
    def __init__(self):
        super().__init__(PreActBlock, [2, 2, 2, 2], num_classes=100)
        self.register_buffer('mu', torch.tensor(
            [0.5070751592371323, 0.48654887331495095, 0.4409178433670343]).view(1, 3, 1, 1))
        self.register_buffer('sigma', torch.tensor(
            [0.2673342858792401, 0.2564384629170883, 0.27615047132568404]).view(1, 3, 1, 1))

    def forward(self, x):
        x = (x - self.mu) / self.sigma
        return super().forward(x)


linf = OrderedDict([
    ('Gowal2020Uncovering', {
        'model': Gowal2020UncoveringNet,
        'gdrive_id': "16I86x2Vv_HCRKROC86G4dQKgO3Po5mT3"
    }),
    ('Gowal2020Uncovering_extra', {
        'model': Gowal2020UncoveringNet,
        'gdrive_id': "1LQBdwO2b391mg7VKcP6I0HIOpC6O83gn"
    }),
    ('Cui2020Learnable_34_20_LBGAT6', {
        'model': lambda: WideResNet(depth=34, widen_factor=20, num_classes=100),
        'gdrive_id': '1rN76st8q_32j6Uo8DI5XhcC2cwVhXBwK'
    }),
    ('Cui2020Learnable_34_10_LBGAT0', {
        'model': lambda: WideResNet(depth=34, widen_factor=10, num_classes=100),
        'gdrive_id': '1RnWbGxN-A-ltsfOvulr68U6i2L8ohAJi'
    }),
    ('Cui2020Learnable_34_10_LBGAT6', {
        'model': lambda: WideResNet(depth=34, widen_factor=10, num_classes=100),
        'gdrive_id': '1TfIgvW3BAkL8jL9J7AAWFSLW3SSzJ2AE'
    }),
    ('Chen2020Efficient', {
        'model': Chen2020EfficientNet,
        'gdrive_id': '1JEh95fvsfKireoELoVCBxOi12IPGFDUT'
    }),
    ('Wu2020Adversarial', {
        'model': Wu2020AdversarialNet,
        'gdrive_id': '1yWGvHmrgjtd9vOpV5zVDqZmeGhCgVYq7'
    }),
    ('Sitawarin2020Improving', {
        'model': lambda: WideResNet(depth=34, widen_factor=10, num_classes=100),
        'gdrive_id': '1hbpwans776KM1SMbOxISkDx0KR0DW8EN'
    }),
    ('Rice2020Overfitting', {
        'model': Rice2020OverfittingNet,
        'gdrive_id': '1XXNZn3fZBOkD1aqNL1cvcD8zZDccyAZ6'
    }),
    ('Hendrycks2019Using', {
        'model': lambda: WideResNet(depth=28, widen_factor=10, num_classes=100),
        'gdrive_id': '1If3tppQsCe5dN8Vbo9ff0tjlKQTTrShd'
    })
])

cifar_100_models = OrderedDict([
    (ThreatModel.Linf, linf)
])

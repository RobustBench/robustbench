from collections import OrderedDict

import torch

from robustbench.model_zoo.architectures.dm_wide_resnet import CIFAR100_MEAN, CIFAR100_STD, \
    DMWideResNet, Swish
from robustbench.model_zoo.architectures.resnet import PreActBlock, PreActResNet
from robustbench.model_zoo.architectures.resnext import CifarResNeXt, ResNeXtBottleneck
from robustbench.model_zoo.architectures.wide_resnet import WideResNet
from robustbench.model_zoo.enums import ThreatModel


class Chen2020EfficientNet(WideResNet):
    def __init__(self, depth=34, widen_factor=10):
        super().__init__(depth=depth,
                         widen_factor=widen_factor,
                         sub_block1=True,
                         num_classes=100)
        self.register_buffer(
            'mu',
            torch.tensor([0.5071, 0.4867, 0.4408]).view(1, 3, 1, 1))
        self.register_buffer(
            'sigma',
            torch.tensor([0.2675, 0.2565, 0.2761]).view(1, 3, 1, 1))

    def forward(self, x):
        x = (x - self.mu) / self.sigma
        return super().forward(x)


class Wu2020AdversarialNet(WideResNet):
    def __init__(self, depth=34, widen_factor=10):
        super().__init__(depth=depth,
                         widen_factor=widen_factor,
                         sub_block1=False,
                         num_classes=100)
        self.register_buffer(
            'mu',
            torch.tensor(
                [0.5070751592371323, 0.48654887331495095,
                 0.4409178433670343]).view(1, 3, 1, 1))
        self.register_buffer(
            'sigma',
            torch.tensor(
                [0.2673342858792401, 0.2564384629170883,
                 0.27615047132568404]).view(1, 3, 1, 1))

    def forward(self, x):
        x = (x - self.mu) / self.sigma
        return super().forward(x)


class Rice2020OverfittingNet(PreActResNet):
    def __init__(self):
        super(Rice2020OverfittingNet, self).__init__(PreActBlock, [2, 2, 2, 2], num_classes=100, bn_before_fc=True, out_shortcut=True)
        self.register_buffer(
            'mu',
            torch.tensor(
                [0.5070751592371323, 0.48654887331495095, 0.4409178433670343]).view(1, 3, 1, 1))
        self.register_buffer(
            'sigma',
            torch.tensor(
                [0.2673342858792401, 0.2564384629170883,
                 0.27615047132568404]).view(1, 3, 1, 1))

    def forward(self, x):
        x = (x - self.mu) / self.sigma
        return super(Rice2020OverfittingNet, self).forward(x)


class Hendrycks2019UsingNet(WideResNet):
    def __init__(self, depth=28, widen_factor=10):
        super(Hendrycks2019UsingNet, self).__init__(depth=depth,
                                                    widen_factor=widen_factor,
                                                    num_classes=100,
                                                    sub_block1=False)

    def forward(self, x):
        x = 2. * x - 1.
        return super(Hendrycks2019UsingNet, self).forward(x)


class Hendrycks2020AugMixResNeXtNet(CifarResNeXt):
    def __init__(self, depth=29, cardinality=4, base_width=32):
        super().__init__(ResNeXtBottleneck,
                         depth=depth,
                         num_classes=100,
                         cardinality=cardinality,
                         base_width=base_width)
        self.register_buffer('mu', torch.tensor([0.5] * 3).view(1, 3, 1, 1))
        self.register_buffer('sigma', torch.tensor([0.5] * 3).view(1, 3, 1, 1))

    def forward(self, x):
        x = (x - self.mu) / self.sigma
        return super().forward(x)


class Hendrycks2020AugMixWRNNet(WideResNet):
    def __init__(self, depth=40, widen_factor=2):
        super().__init__(depth=depth,
                         widen_factor=widen_factor,
                         sub_block1=False,
                         num_classes=100)
        self.register_buffer('mu', torch.tensor([0.5] * 3).view(1, 3, 1, 1))
        self.register_buffer('sigma', torch.tensor([0.5] * 3).view(1, 3, 1, 1))

    def forward(self, x):
        x = (x - self.mu) / self.sigma
        return super().forward(x)


linf = OrderedDict([
    ('Gowal2020Uncovering', {
        'model':
        lambda: DMWideResNet(num_classes=100,
                             depth=70,
                             width=16,
                             activation_fn=Swish,
                             mean=CIFAR100_MEAN,
                             std=CIFAR100_STD),
        'gdrive_id':
        "16I86x2Vv_HCRKROC86G4dQKgO3Po5mT3"
    }),
    ('Gowal2020Uncovering_extra', {
        'model':
        lambda: DMWideResNet(num_classes=100,
                             depth=70,
                             width=16,
                             activation_fn=Swish,
                             mean=CIFAR100_MEAN,
                             std=CIFAR100_STD),
        'gdrive_id':
        "1LQBdwO2b391mg7VKcP6I0HIOpC6O83gn"
    }),
    ('Cui2020Learnable_34_20_LBGAT6', {
        'model':
        lambda: WideResNet(
            depth=34, widen_factor=20, num_classes=100, sub_block1=True),
        'gdrive_id':
        '1rN76st8q_32j6Uo8DI5XhcC2cwVhXBwK'
    }),
    ('Cui2020Learnable_34_10_LBGAT0', {
        'model':
        lambda: WideResNet(
            depth=34, widen_factor=10, num_classes=100, sub_block1=True),
        'gdrive_id':
        '1RnWbGxN-A-ltsfOvulr68U6i2L8ohAJi'
    }),
    ('Cui2020Learnable_34_10_LBGAT6', {
        'model':
        lambda: WideResNet(
            depth=34, widen_factor=10, num_classes=100, sub_block1=True),
        'gdrive_id':
        '1TfIgvW3BAkL8jL9J7AAWFSLW3SSzJ2AE'
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
        'model':
        lambda: WideResNet(
            depth=34, widen_factor=10, num_classes=100, sub_block1=True),
        'gdrive_id':
        '1hbpwans776KM1SMbOxISkDx0KR0DW8EN'
    }),
    ('Hendrycks2019Using', {
        'model': Hendrycks2019UsingNet, 
        'gdrive_id': '1If3tppQsCe5dN8Vbo9ff0tjlKQTTrShd'
    }),
    ('Rice2020Overfitting', {
        'model': Rice2020OverfittingNet,
        'gdrive_id': '1XXNZn3fZBOkD1aqNL1cvcD8zZDccyAZ6'
    }),
    ('Rebuffi2021Fixing_70_16_cutmix_ddpm', {
        'model':
        lambda: DMWideResNet(num_classes=100,
                             depth=70,
                             width=16,
                             activation_fn=Swish,
                             mean=CIFAR100_MEAN,
                             std=CIFAR100_STD),
        'gdrive_id': '1-GkVLo9QaRjCJl-by67xda1ySVhYxsLV'
    }),
    ('Rebuffi2021Fixing_28_10_cutmix_ddpm', {
        'model':
        lambda: DMWideResNet(num_classes=100,
                             depth=28,
                             width=10,
                             activation_fn=Swish,
                             mean=CIFAR100_MEAN,
                             std=CIFAR100_STD),
        'gdrive_id': '1-P7cs82Tj6UVx7Coin3tVurVKYwXWA9p'
    }),
])

common_corruptions = OrderedDict([
    ('Hendrycks2020AugMix_WRN', {
        'model': Hendrycks2020AugMixWRNNet,
        'gdrive_id': '1XpFFdCdU9LcDtcyNfo6_BV1RZHKKkBVE'
    }),
    ('Hendrycks2020AugMix_ResNeXt', {
      'model': Hendrycks2020AugMixResNeXtNet,
      'gdrive_id': '1ocnHbvDdOBLvgNr6K7vEYL08hUdkD1Rv'
    })
])

cifar_100_models = OrderedDict([(ThreatModel.Linf, linf),
                                (ThreatModel.corruptions, common_corruptions)])

from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

from robustbench.model_zoo.architectures.dm_wide_resnet import CIFAR10_MEAN, CIFAR10_STD, \
    DMWideResNet, Swish
from robustbench.model_zoo.architectures.resnet import Bottleneck, BottleneckChen2020AdversarialNet, \
    PreActBlock, \
    PreActBlockV2, PreActResNet, ResNet
from robustbench.model_zoo.architectures.resnext import CifarResNeXt, \
    ResNeXtBottleneck
from robustbench.model_zoo.architectures.wide_resnet import WideResNet
from robustbench.model_zoo.enums import ThreatModel


class Hendrycks2020AugMixResNeXtNet(CifarResNeXt):
    def __init__(self, depth=29, num_classes=10, cardinality=4, base_width=32):
        super().__init__(ResNeXtBottleneck, depth=depth, num_classes=num_classes,
                         cardinality=cardinality, base_width=base_width)


class Hendrycks2020AugMixWRNNet(WideResNet):
    def __init__(self, depth=40, widen_factor=2):
        super().__init__(depth=depth, widen_factor=widen_factor, sub_block1=False)


class Gowal2020UncoveringNet(DMWideResNet):
    def __init__(self, depth=70, width=16):
        super().__init__(depth=depth, width=width, activation_fn=Swish, mean=CIFAR10_MEAN,
                         std=CIFAR10_STD)


class Carmon2019UnlabeledNet(WideResNet):
    def __init__(self, depth=28, widen_factor=10):
        super(Carmon2019UnlabeledNet, self).__init__(depth=depth, widen_factor=widen_factor,
                                                     sub_block1=True)


class Sehwag2020PruningNet(WideResNet):
    def __init__(self, depth=28, widen_factor=10):
        super(Sehwag2020PruningNet, self).__init__(depth=depth, widen_factor=widen_factor,
                                                   sub_block1=True)


class Wang2020ImprovingNet(WideResNet):
    def __init__(self, depth=28, widen_factor=10):
        super(Wang2020ImprovingNet, self).__init__(depth=depth, widen_factor=widen_factor,
                                                   sub_block1=True)


class Hendrycks2019UsingNet(WideResNet):
    def __init__(self, depth=28, widen_factor=10):
        super(Hendrycks2019UsingNet, self).__init__(depth=depth, widen_factor=widen_factor,
                                                    sub_block1=False)

    def forward(self, x):
        x = 2. * x - 1.
        return super(Hendrycks2019UsingNet, self).forward(x)


class Rice2020OverfittingNet(WideResNet):
    def __init__(self, depth=34, widen_factor=20):
        super(Rice2020OverfittingNet, self).__init__(depth=depth, widen_factor=widen_factor,
                                                     sub_block1=False)
        self.register_buffer('mu', torch.tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1))
        self.register_buffer('sigma', torch.tensor([0.2471, 0.2435, 0.2616]).view(1, 3, 1, 1))

    def forward(self, x):
        x = (x - self.mu) / self.sigma
        return super(Rice2020OverfittingNet, self).forward(x)


class Zhang2019TheoreticallyNet(WideResNet):
    def __init__(self, depth=34, widen_factor=10):
        super(Zhang2019TheoreticallyNet, self).__init__(depth=depth, widen_factor=widen_factor,
                                                        sub_block1=True)


class Engstrom2019RobustnessNet(ResNet):
    def __init__(self):
        super(Engstrom2019RobustnessNet, self).__init__(Bottleneck, [3, 4, 6, 3])
        self.register_buffer('mu', torch.tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1))
        self.register_buffer('sigma', torch.tensor([0.2023, 0.1994, 0.2010]).view(1, 3, 1, 1))

    def forward(self, x):
        x = (x - self.mu) / self.sigma
        return super(Engstrom2019RobustnessNet, self).forward(x)


class Chen2020AdversarialNet(torch.nn.Module):
    def __init__(self):
        super(Chen2020AdversarialNet, self).__init__()
        self.branch1 = ResNet(BottleneckChen2020AdversarialNet, [3, 4, 6, 3])
        self.branch2 = ResNet(BottleneckChen2020AdversarialNet, [3, 4, 6, 3])
        self.branch3 = ResNet(BottleneckChen2020AdversarialNet, [3, 4, 6, 3])

        self.models = [self.branch1, self.branch2, self.branch3]

        self.register_buffer('mu', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('sigma', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x):
        out = (x - self.mu) / self.sigma

        out1 = self.branch1(out)
        out2 = self.branch2(out)
        out3 = self.branch3(out)

        prob1 = torch.softmax(out1, dim=1)
        prob2 = torch.softmax(out2, dim=1)
        prob3 = torch.softmax(out3, dim=1)

        return (prob1 + prob2 + prob3) / 3


class Huang2020SelfNet(WideResNet):
    def __init__(self, depth=34, widen_factor=10):
        super(Huang2020SelfNet, self).__init__(depth=depth, widen_factor=widen_factor,
                                               sub_block1=True)


class Pang2020BoostingNet(WideResNet):
    def __init__(self, depth=34, widen_factor=20):
        super(Pang2020BoostingNet, self).__init__(depth=depth, widen_factor=widen_factor,
                                                  sub_block1=True, bias_last=False)
        self.register_buffer('mu', torch.tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1))
        self.register_buffer('sigma', torch.tensor([0.2471, 0.2435, 0.2616]).view(1, 3, 1, 1))

    def forward(self, x):
        x = (x - self.mu) / self.sigma
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(-1, self.nChannels)
        out = F.normalize(out, p=2, dim=1)
        for _, module in self.fc.named_modules():
            if isinstance(module, nn.Linear):
                module.weight.data = F.normalize(module.weight, p=2, dim=1)
        return self.fc(out)


class Wong2020FastNet(PreActResNet):
    def __init__(self):
        super(Wong2020FastNet, self).__init__(PreActBlock, [2, 2, 2, 2])
        self.register_buffer('mu', torch.tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1))
        self.register_buffer('sigma', torch.tensor([0.2471, 0.2435, 0.2616]).view(1, 3, 1, 1))

    def forward(self, x):
        x = (x - self.mu) / self.sigma
        return super(Wong2020FastNet, self).forward(x)


class Ding2020MMANet(WideResNet):
    """
    See the appendix of the LICENSE file specifically for this model.
    """

    def __init__(self, depth=28, widen_factor=4):
        super(Ding2020MMANet, self).__init__(depth=depth, widen_factor=widen_factor,
                                             sub_block1=False)

    def forward(self, x):
        mu = x.mean(dim=(1, 2, 3), keepdim=True)
        std = x.std(dim=(1, 2, 3), keepdim=True)
        std_min = torch.ones_like(std) / (x.shape[1] * x.shape[2] * x.shape[3]) ** .5
        x = (x - mu) / torch.max(std, std_min)
        return super(Ding2020MMANet, self).forward(x)


class Zhang2019YouNet(WideResNet):
    def __init__(self, depth=34, widen_factor=10):
        super(Zhang2019YouNet, self).__init__(depth=depth, widen_factor=widen_factor,
                                              sub_block1=True)


class StandardNet(WideResNet):
    def __init__(self, depth=28, widen_factor=10):
        super(StandardNet, self).__init__(depth=depth, widen_factor=widen_factor, sub_block1=False)


class Zhang2020AttacksNet(WideResNet):
    def __init__(self, depth=34, widen_factor=10):
        super(Zhang2020AttacksNet, self).__init__(depth=depth, widen_factor=widen_factor,
                                                  sub_block1=True)


class Augustin2020AdversarialNet(ResNet):
    def __init__(self):
        super(Augustin2020AdversarialNet, self).__init__(Bottleneck, [3, 4, 6, 3])
        self.register_buffer('mu',
                             torch.tensor([0.4913997551666284, 0.48215855929893703,
                                           0.4465309133731618]).view(1, 3, 1,
                                                                     1))
        self.register_buffer('sigma',
                             torch.tensor(
                                 [0.24703225141799082, 0.24348516474564, 0.26158783926049628]).view(
                                 1, 3, 1,
                                 1))

    def forward(self, x):
        x = (x - self.mu) / self.sigma
        return super(Augustin2020AdversarialNet, self).forward(x)


class Rice2020OverfittingNetL2(PreActResNet):
    def __init__(self):
        super(Rice2020OverfittingNetL2, self).__init__(PreActBlockV2, [2, 2, 2, 2],
                                                       bn_before_fc=True)
        self.register_buffer('mu', torch.tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1))
        self.register_buffer('sigma', torch.tensor([0.2471, 0.2435, 0.2616]).view(1, 3, 1, 1))

    def forward(self, x):
        x = (x - self.mu) / self.sigma
        return super(Rice2020OverfittingNetL2, self).forward(x)


class Rony2019DecouplingNet(WideResNet):
    def __init__(self, depth=28, widen_factor=10):
        super(Rony2019DecouplingNet, self).__init__(depth=depth, widen_factor=widen_factor,
                                                    sub_block1=False)
        self.register_buffer('mu', torch.tensor([0.491, 0.482, 0.447]).view(1, 3, 1, 1))
        self.register_buffer('sigma', torch.tensor([0.247, 0.243, 0.262]).view(1, 3, 1, 1))

    def forward(self, x):
        x = (x - self.mu) / self.sigma
        return super(Rony2019DecouplingNet, self).forward(x)


class Wu2020AdversarialNet(WideResNet):
    def __init__(self, depth=28, widen_factor=10):
        super(Wu2020AdversarialNet, self).__init__(depth=depth, widen_factor=widen_factor,
                                                   sub_block1=True)


class Wu2020AdversarialNetL2(WideResNet):
    def __init__(self, depth=34, widen_factor=10):
        super(Wu2020AdversarialNetL2, self).__init__(depth=depth, widen_factor=widen_factor,
                                                     sub_block1=False)


linf = OrderedDict([
    ('Carmon2019Unlabeled', {
        'model': Carmon2019UnlabeledNet,
        'gdrive_id': '15tUx-gkZMYx7BfEOw1GY5OKC-jECIsPQ',
    }),
    ('Sehwag2020Hydra', {
        'model': Sehwag2020PruningNet,
        'gdrive_id': '1pi8GHwAVkxVH41hEnf0IAJb_7y-Q8a2Y',
    }),
    ('Wang2020Improving', {
        'model': Wang2020ImprovingNet,
        'gdrive_id': '1T939mU4kXYt5bbvM55aT4fLBvRhyzjiQ',
    }),
    ('Hendrycks2019Using', {
        'model': Hendrycks2019UsingNet,
        'gdrive_id': '1-DcJsYw2dNEOyF9epks2QS7r9nqBDEsw',
    }),
    ('Rice2020Overfitting', {
        'model': Rice2020OverfittingNet,
        'gdrive_id': '1vC_Twazji7lBjeMQvAD9uEQxi9Nx2oG-',
    }),
    ('Zhang2019Theoretically', {
        'model': Zhang2019TheoreticallyNet,
        'gdrive_id': '1hPz9QQwyM7QSuWu-ANG_uXR-29xtL8t_',
    }),
    ('Engstrom2019Robustness', {
        'model': Engstrom2019RobustnessNet,
        'gdrive_id': '1etqmQsksNIWBvBQ4r8ZFk_3FJlLWr8Rr',
    }),
    ('Chen2020Adversarial', {
        'model': Chen2020AdversarialNet,
        'gdrive_id': ['1HrG22y_A9F0hKHhh2cLLvKxsQTJTLE_y',
                      '1DB2ymt0rMnsMk5hTuUzoMTpMKEKWpExd',
                      '1GfgzNZcC190-IrT7056IZFDB6LfMUL9m'],
    }),
    ('Huang2020Self', {
        'model': Huang2020SelfNet,
        'gdrive_id': '1nInDeIyZe2G-mJFxQJ3UoclQNomWjMgm',
    }),
    ('Pang2020Boosting', {
        'model': Pang2020BoostingNet,
        'gdrive_id': '1iNWOj3MP7kGe8yTAS4XnDaDXDLt0mwqw',
    }),
    ('Wong2020Fast', {
        'model': Wong2020FastNet,
        'gdrive_id': '1Re--_lf3jCEw9bnQqGkjw3J7v2tSZKrv',
    }),
    ('Ding2020MMA', {
        'model': Ding2020MMANet,
        'gdrive_id': '19Q_rIIHXsYzxZ0WcZdqT-N2OD7MfgoZ0',
    }),
    ('Zhang2019You', {
        'model': Zhang2019YouNet,
        'gdrive_id': '1kB2qqPQ8qUNmK8VKuTOhT1X4GT46kAoA',
    }),
    ('Standard', {
        'model': StandardNet,
        'gdrive_id': '1t98aEuzeTL8P7Kpd5DIrCoCL21BNZUhC',
    }),
    ('Zhang2020Attacks', {
        'model': Zhang2020AttacksNet,
        'gdrive_id': '1lBVvLG6JLXJgQP2gbsTxNHl6s3YAopqk',
    }),
    ('Wu2020Adversarial_extra', {
        'model': Wu2020AdversarialNet,
        'gdrive_id': '1-WJWpAZLlmc4gJ8XXNf7IETjnSZzaCNp',
    }),
    ('Wu2020Adversarial', {
        'model': Wu2020AdversarialNetL2,
        'gdrive_id': '13LBcgNvhFppCFG22i1xATrahFPfMgXGf',
    }),
    ('Gowal2020Uncovering', {
        'model': Gowal2020UncoveringNet,
        'gdrive_id': "11EDXB3pef30mtiJpW46uFNb21udQFAt5"
    }),
    ('Gowal2020Uncovering_extra', {
        'model': Gowal2020UncoveringNet,
        'gdrive_id': "1HSwu57j90iuLinviH-9m5uClEyZedbGl"
    })
])

l2 = OrderedDict([
    ('Augustin2020Adversarial', {
        'model': Augustin2020AdversarialNet,
        'gdrive_id': '1oDghrzNfkStC2wr5Fq8T896yNV4wVG4d',
    }),
    ('Engstrom2019Robustness', {
        'model': Engstrom2019RobustnessNet,
        'gdrive_id': '1O8rGa6xOUIRwQ-M4ESrCjzknby8TM2ZE',
    }),
    ('Rice2020Overfitting', {
        'model': Rice2020OverfittingNetL2,
        'gdrive_id': '1jo-31utiYNBVzLM0NxUEWz0teo3Z0xa7',
    }),
    ('Rony2019Decoupling', {
        'model': Rony2019DecouplingNet,
        'gdrive_id': '1Oua2ZYSxNvoDrtlY9vTtRzyBWHziE4Uy',
    }),
    ('Standard', {
        'model': StandardNet,
        'gdrive_id': '1t98aEuzeTL8P7Kpd5DIrCoCL21BNZUhC',
    }),
    ('Ding2020MMA', {
        'model': Ding2020MMANet,
        'gdrive_id': '13wgY0Q_eor52ltZ0PkfJx5BCZ8cLM52E',
    }),
    ('Wu2020Adversarial', {
        'model': Wu2020AdversarialNetL2,
        'gdrive_id': '1M5AZ0EZQt7d2AlTmsnqZcfx91-x7YEAV',
    }),
    ('Gowal2020Uncovering', {
        'model': Gowal2020UncoveringNet,
        'gdrive_id': "1MxJEYcACdbg8ygmW4wNZzfajZrbeecG6"
    }),
    ('Gowal2020Uncovering_extra', {
        'model': Gowal2020UncoveringNet,
        'gdrive_id': "17RwGZ1uXcWp-Vndnv6YVC9eRcHLtV6Hk"
    })
])

common_corruptions = OrderedDict([
    ('Hendrycks2020AugMixWRN', {
        'model': Hendrycks2020AugMixWRNNet,
        'gdrive_id': "1wy7gSRsUZzCzj8QhmTbcnwmES_2kkNph"
    }),
    ('Hendrycks2020AugMixResNeXt', {
        'model': Hendrycks2020AugMixResNeXtNet,
        'gdrive_id': "1uGP3nZbL3LC160kOsxwkkt6tDd4qbZT1"
    })
])

cifar_10_models = OrderedDict([
    (ThreatModel.Linf, linf),
    (ThreatModel.L2, l2),
    (ThreatModel.corruptions, common_corruptions)
])

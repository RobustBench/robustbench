import torch
from collections import OrderedDict
from model_zoo.wide_resnet import WideResNet
from model_zoo.resnet import ResNet, Bottleneck


class Carmon2019UnlabeledNet(WideResNet):
    def __init__(self, depth, widen_factor):
        super(Carmon2019UnlabeledNet, self).__init__(depth=depth, widen_factor=widen_factor, sub_block1=True)


class Sehwag2020PruningNet(WideResNet):
    def __init__(self, depth, widen_factor):
        super(Sehwag2020PruningNet, self).__init__(depth=depth, widen_factor=widen_factor, sub_block1=True)


class Wang2020ImprovingNet(WideResNet):
    def __init__(self, depth, widen_factor):
        super(Wang2020ImprovingNet, self).__init__(depth=depth, widen_factor=widen_factor, sub_block1=True)


class Hendrycks2019UsingNet(WideResNet):
    def __init__(self, depth, widen_factor):
        super(Hendrycks2019UsingNet, self).__init__(depth=depth, widen_factor=widen_factor, sub_block1=False)

    def forward(self, x, return_prelogit=False):
        x = 2. * x - 1.
        return super(Hendrycks2019UsingNet, self).forward(x)


class Rice2020OverfittingNet(WideResNet):
    def __init__(self, depth, widen_factor):
        super(Rice2020OverfittingNet, self).__init__(depth=depth, widen_factor=widen_factor, sub_block1=False)

    def forward(self, x, return_prelogit=False):
        mu = torch.Tensor([0.4914, 0.4822, 0.4465]).float().view(3, 1, 1).cuda()
        sigma = torch.Tensor([0.2471, 0.2435, 0.2616]).float().view(3, 1, 1).cuda()
        x = (x - mu) / sigma
        return super(Rice2020OverfittingNet, self).forward(x)


class Zhang2019TheoreticallyNet(WideResNet):
    def __init__(self, depth, widen_factor):
        super(Zhang2019TheoreticallyNet, self).__init__(depth=depth, widen_factor=widen_factor, sub_block1=True)


class Engstrom2019RobustnessNet(ResNet):
    def __init__(self):
        super(Engstrom2019RobustnessNet, self).__init__(Bottleneck, [3, 4, 6, 3])  # ResNet-50


model_dicts = OrderedDict([
    ('Carmon2019Unlabeled', {
        'model': Carmon2019UnlabeledNet(28, 10),
        'gdrive_id': '15tUx-gkZMYx7BfEOw1GY5OKC-jECIsPQ',
    }),
    ('Sehwag2020Hydra', {
        'model': Sehwag2020PruningNet(28, 10),
        'gdrive_id': '1pi8GHwAVkxVH41hEnf0IAJb_7y-Q8a2Y',
    }),
    ('Wang2020Improving', {
        'model': Wang2020ImprovingNet(28, 10),
        'gdrive_id': '1T939mU4kXYt5bbvM55aT4fLBvRhyzjiQ',
    }),
    ('Hendrycks2019Using', {
        'model': Hendrycks2019UsingNet(28, 10),
        'gdrive_id': '1-DcJsYw2dNEOyF9epks2QS7r9nqBDEsw',
    }),
    ('Rice2020Overfitting', {
        'model': Rice2020OverfittingNet(34, 20),
        'gdrive_id': '1vC_Twazji7lBjeMQvAD9uEQxi9Nx2oG-',
    }),
    ('Zhang2019Theoretically', {
        'model': Zhang2019TheoreticallyNet(34, 10),
        'gdrive_id': '1hPz9QQwyM7QSuWu-ANG_uXR-29xtL8t_',
    }),
    ('Engstrom2019Robustness', {
        'model': Engstrom2019RobustnessNet(),
        'gdrive_id': '1etqmQsksNIWBvBQ4r8ZFk_3FJlLWr8Rr',
    }),
])

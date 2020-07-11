import torch
from collections import OrderedDict
from model_zoo.wide_resnet import WideResNet
from model_zoo.resnet import ResNet, Bottleneck
from model_zoo.resnetv2 import ResNet as ResNetv2, Bottleneck as Bottleneckv2


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

    def forward(self, x):
        x = 2. * x - 1.
        return super(Hendrycks2019UsingNet, self).forward(x)


class Rice2020OverfittingNet(WideResNet):
    def __init__(self, depth, widen_factor):
        super(Rice2020OverfittingNet, self).__init__(depth=depth, widen_factor=widen_factor, sub_block1=False)
        self.mu = torch.Tensor([0.4914, 0.4822, 0.4465]).float().view(3, 1, 1).cuda()
        self.sigma = torch.Tensor([0.2471, 0.2435, 0.2616]).float().view(3, 1, 1).cuda()
    
    def forward(self, x):
        x = (x - self.mu) / self.sigma
        return super(Rice2020OverfittingNet, self).forward(x)


class Zhang2019TheoreticallyNet(WideResNet):
    def __init__(self, depth, widen_factor):
        super(Zhang2019TheoreticallyNet, self).__init__(depth=depth, widen_factor=widen_factor, sub_block1=True)


class Engstrom2019RobustnessNet(ResNet):
    def __init__(self):
        super(Engstrom2019RobustnessNet, self).__init__(Bottleneck, [3, 4, 6, 3])
        self.mu = torch.Tensor([0.4914, 0.4822, 0.4465]).float().view(3, 1, 1).cuda()
        self.sigma = torch.Tensor([0.2023, 0.1994, 0.2010]).float().view(3, 1, 1).cuda()
    
    def forward(self, x):
        x = (x - self.mu) / self.sigma
        return super(Engstrom2019RobustnessNet, self).forward(x)


class Chen2020AdversarialNet(torch.nn.Module):
    def __init__(self):
        super(Chen2020AdversarialNet, self).__init__()
        self.branch1 = ResNetv2(Bottleneckv2, [3, 4, 6, 3])
        self.branch2 = ResNetv2(Bottleneckv2, [3, 4, 6, 3])
        self.branch3 = ResNetv2(Bottleneckv2, [3, 4, 6, 3])
        
        self.models = [self.branch1, self.branch2, self.branch3]

        self.mu = torch.Tensor([0.485, 0.456, 0.406]).float().view(1, 3, 1, 1).cuda()
        self.sigma = torch.Tensor([0.229, 0.224, 0.225]).float().view(1, 3, 1, 1).cuda()
    
    def forward(self, x):
        out = (x - self.mu) / self.sigma
        
        out1 = self.branch1(out)
        out2 = self.branch2(out)
        out3 = self.branch3(out)
        
        logit1 = torch.softmax(out1, dim=1)
        logit2 = torch.softmax(out2, dim=1)
        logit3 = torch.softmax(out3, dim=1)

        return (logit1 + logit2 + logit3) / 3
            

class Huang2020SelfNet(WideResNet):
    def __init__(self, depth, widen_factor):
        super(Huang2020SelfNet, self).__init__(depth=depth, widen_factor=widen_factor, sub_block1=True)


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
    ('Chen2020Adversarial', {
        'model': Chen2020AdversarialNet(),
        'gdrive_id': ['1HrG22y_A9F0hKHhh2cLLvKxsQTJTLE_y',
            '1DB2ymt0rMnsMk5hTuUzoMTpMKEKWpExd',
            '1GfgzNZcC190-IrT7056IZFDB6LfMUL9m'],
    }),
    ('Huang2020Self', {
        'model': Huang2020SelfNet(34, 10),
        'gdrive_id': 't',
    }),
])

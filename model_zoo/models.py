from model_zoo.model_base import WideResNet, WideResNetV2, WideResNetV3
from model_zoo.resnet import ResNet50
#import torch.nn

class Carmon2019UnlabeledNet(WideResNet):
    def __init__(self, depth, widen_factor):
        super(Carmon2019UnlabeledNet, self).__init__(depth=depth, widen_factor=widen_factor)

class Sehwag2020PruningNet(WideResNet):
    def __init__(self, depth, widen_factor):
        super(Sehwag2020PruningNet, self).__init__(depth=depth, widen_factor=widen_factor)

class Wang2020ImprovingNet(WideResNet):
    def __init__(self, depth, widen_factor):
        super(Wang2020ImprovingNet, self).__init__(depth=depth, widen_factor=widen_factor)

class Hendrycks2019UsingNet(WideResNetV2):
    def __init__(self, depth, widen_factor):
        super(Hendrycks2019UsingNet, self).__init__(depth=depth, widen_factor=widen_factor)

class Rice2020OverfittingNet(WideResNetV3):
    def __init__(self, depth, widen_factor):
        super(Rice2020OverfittingNet, self).__init__(depth=depth, widen_factor=widen_factor)

class Zhang2019TheoreticallyNet(WideResNet):
    def __init__(self, depth, widen_factor):
        super(Zhang2019TheoreticallyNet, self).__init__(depth=depth, widen_factor=widen_factor)

'''class Engstrom2019RobustnessNet(ResNet50):
    def __init__(self):
        super(Engstrom2019RobustnessNet, self).__init__()'''

model_dicts = {
    'carmon2019unlabeled': {
        'model': Carmon2019UnlabeledNet(28, 10),
        'gdrive_id': '15tUx-gkZMYx7BfEOw1GY5OKC-jECIsPQ',
        'ckpt_ext': 'pt'
    },
    'sehwag2020pruning': {
        'model': Sehwag2020PruningNet(28, 10),
        'gdrive_id': '1pi8GHwAVkxVH41hEnf0IAJb_7y-Q8a2Y',
        'ckpt_ext': 'tar'
    },
    'wang2020improving': {
        'model': Wang2020ImprovingNet(28, 10),
        'gdrive_id': '1T939mU4kXYt5bbvM55aT4fLBvRhyzjiQ',
        'ckpt_ext': 'pt'
    },
    'hendrycks2019using': {
        'model': Hendrycks2019UsingNet(28, 10),
        'gdrive_id': '1-DcJsYw2dNEOyF9epks2QS7r9nqBDEsw',
        'ckpt_ext': 'pt'
    },
    'rice2020overfitting': {
        'model': Rice2020OverfittingNet(34, 20),
        'gdrive_id': '1vC_Twazji7lBjeMQvAD9uEQxi9Nx2oG-',
        'ckpt_ext': 'pth'
    },
    'zhang2019theoretically': {
        'model': Zhang2019TheoreticallyNet(34, 10),
        'gdrive_id': '1hPz9QQwyM7QSuWu-ANG_uXR-29xtL8t_',
        'ckpt_ext': 'pt'
    },
    'engstrom2019robustness': {
        'model': ResNet50(), #Engstrom2019RobustnessNet()
        'gdrive_id': '1etqmQsksNIWBvBQ4r8ZFk_3FJlLWr8Rr',
        'ckpt_ext': 'pt'
    }
}#

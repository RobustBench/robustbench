from model_zoo.model_base import WideResNet, WideResNetV2, WideResNetV3
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

model_dicts = {
    'carmon2019unlabeled': {
        'model': Carmon2019UnlabeledNet(28, 10),
        'gdrive_id': '15tUx-gkZMYx7BfEOw1GY5OKC-jECIsPQ',
        'ckpt_ext': 'pt'
    },
    'sehwag2020pruning': {
        'model': Sehwag2020PruningNet(28, 10),
        'gdrive_id': '1c8NUE5y3-qV6_AVLic5tvm4T_zqASPIN',
        'ckpt_ext': 'tar'
    },
    'wang2020improving': {
        'model': Wang2020ImprovingNet(28, 10),
        'gdrive_id': '17wsyyPSRcuu4k8nrqOd0Dvda28PBM0mg',
        'ckpt_ext': 'pt'
    },
    'hendrycks2019using': {
        'model': Hendrycks2019UsingNet(28, 10),
        'gdrive_id': '1FNYZoIow9RpD3UmSEhPqNuTOvRRlVfsp', #'1MILK-czYzJqjaRfSlSEnuF_5q5F52Bw9'
        'ckpt_ext': 'pt'
    },
    'rice2020overfitting': {
        'model': Rice2020OverfittingNet(34, 20),
        'gdrive_id': '12LhJ4Y3jxOrLS87ktBmIMlnX4pkkMZmQ',
        'ckpt_ext': 'pt'
    }
}#

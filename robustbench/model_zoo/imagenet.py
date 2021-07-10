from collections import OrderedDict

import torch

#from robustbench.model_zoo.architectures.resnet import Bottleneck, ResNet18
from torchvision import models as pt_models

from robustbench.model_zoo.enums import ThreatModel
from robustbench.model_zoo.architectures.utils_architectures import normalize_model


mu = [0.485, 0.456, 0.406]
sigma = [0.229, 0.224, 0.225]


linf = OrderedDict(
    [
        ('Wong2020Fast', { # requires resolution 288 x 288
            'model': lambda: normalize_model(pt_models.resnet50(
                pretrained=False), mu, sigma),
            'gdrive_id': ''
        }),
        ('Engstrom2019Robustness', {
            'model': lambda: normalize_model(pt_models.resnet50(), mu, sigma),
            'gdrive_id': ''
        }),
        ('Salman2020Do_R50', {
            'model': lambda: normalize_model(pt_models.resnet50(), mu, sigma),
            'gdrive_id': ''
        }),
        ('Salman2020Do_R18', {
            'model': lambda: normalize_model(pt_models.resnet18(), mu, sigma),
            'gdrive_id': ''
        }),
        ('Salman2020Do_50_2', {
            'model': lambda: normalize_model(pt_models.wide_resnet50_2(),
                mu, sigma),
            'gdrive_id': ''
        }),
    ])

imagenet_models = OrderedDict([(ThreatModel.Linf, linf)
    ])



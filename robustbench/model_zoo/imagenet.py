from collections import OrderedDict

from torchvision import models as pt_models

from robustbench.model_zoo.enums import ThreatModel
from robustbench.model_zoo.architectures.utils_architectures import normalize_model


mu = (0.485, 0.456, 0.406)
sigma = (0.229, 0.224, 0.225)


linf = OrderedDict(
    [
        ('Wong2020Fast', {  # requires resolution 288 x 288
            'model': lambda: normalize_model(pt_models.resnet50(), mu, sigma),
            'gdrive_id': '1deM2ZNS5tf3S_-eRURJi-IlvUL8WJQ_w',
            'preprocessing': 'Crop288'
        }),
        ('Engstrom2019Robustness', {
            'model': lambda: normalize_model(pt_models.resnet50(), mu, sigma),
            'gdrive_id': '1T2Fvi1eCJTeAOEzrH_4TAIwO8HTOYVyn',
            'preprocessing': 'Res256Crop224',
        }),
        ('Salman2020Do_R50', {
            'model': lambda: normalize_model(pt_models.resnet50(), mu, sigma),
            'gdrive_id': '1TmT5oGa1UvVjM3d-XeSj_XmKqBNRUg8r',
            'preprocessing': 'Res256Crop224'
        }),
        ('Salman2020Do_R18', {
            'model': lambda: normalize_model(pt_models.resnet18(), mu, sigma),
            'gdrive_id': '1OThCOQCOxY6lAgxZxgiK3YuZDD7PPfPx',
            'preprocessing': 'Res256Crop224'
        }),
        ('Salman2020Do_50_2', {
            'model': lambda: normalize_model(pt_models.wide_resnet50_2(), mu, sigma),
            'gdrive_id': '1OT7xaQYljrTr3vGbM37xK9SPoPJvbSKB',
            'preprocessing': 'Res256Crop224'
        }),
        ('Standard_R50', {
            'model': lambda: normalize_model(pt_models.resnet50(pretrained=True), mu, sigma),
            'gdrive_id': '',
            'preprocessing': 'Res256Crop224'
        }),
    ])

imagenet_models = OrderedDict([(ThreatModel.Linf, linf)])



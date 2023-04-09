from typing import Optional, Tuple
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.helpers import build_model_with_cfg
from timm.models.registry import register_model
from torchvision.models.alexnet import AlexNet

from .utils_architectures import normalize_timm_model


def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000,
        'input_size': (3, 224, 224),
        'pool_size': None,
        'crop_pct': 1.0,
        'interpolation': 'bilinear',
        'fixed_input_size': True,
        'mean': IMAGENET_DEFAULT_MEAN,
        'std': IMAGENET_DEFAULT_STD,
        'first_conv': None,
        'classifier': 'classifier',
        **kwargs
    }


default_cfgs = {
    'alexnet_imagenet_corruptions':
    _cfg(url="https://download.pytorch.org/models/alexnet-owt-7be5be79.pth")
}


class TimmAlexNet(AlexNet):
    def __init__(self,
                 num_classes: int = 1000,
                 dropout: float = 0.5,
                 in_chans: Optional[int] = None,
                 img_size: Optional[Tuple[int]] = None) -> None:
        super().__init__(num_classes, dropout)


def _create_alexnet(variant, pretrained=False, default_cfg=None, **kwargs):
    model = build_model_with_cfg(TimmAlexNet, variant, pretrained, **kwargs)
    return model


@register_model
def alexnet_imagenet_corruptions(pretrained=False, **kwargs):
    model_kwargs = dict(**kwargs)
    model = _create_alexnet('alexnet_imagenet_corruptions',
                            pretrained=pretrained,
                            **model_kwargs)
    assert isinstance(model, TimmAlexNet)
    model = normalize_timm_model(model)
    return model


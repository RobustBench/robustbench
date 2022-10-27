from timm.models import deit
from timm.models.registry import register_model

from .utils_architectures import normalize_timm_model

default_cfgs = {
    'tian2022deepeer_deit_s_imagenet_corruptions':
    deit._cfg(
        url=
        "https://github.com/RobustBench/robustbench/releases/download/v1.2/tian2022deeper-deit-s.pth"
    ),
    'tian2022deepeer_deit_b_imagenet_corruptions':
    deit._cfg(
        url=
        "https://github.com/RobustBench/robustbench/releases/download/v1.2/tian2022deeper-deit-b.pth"
    ),
}


@register_model
def tian2022deepeer_deit_s_imagenet_corruptions(pretrained=False,
                           **kwargs) -> deit.VisionTransformer:
    model_kwargs = dict(patch_size=16,
                        embed_dim=384,
                        depth=12,
                        num_heads=6,
                        **kwargs)
    model = deit._create_deit('tian2022deepeer_deit_s_imagenet_corruptions',
                              pretrained=pretrained,
                              **model_kwargs)
    assert isinstance(model, deit.VisionTransformer)
    model = normalize_timm_model(model)
    return model


@register_model
def tian2022deepeer_deit_b_imagenet_corruptions(pretrained=False,
                           **kwargs) -> deit.VisionTransformer:
    model_kwargs = dict(patch_size=16,
                        embed_dim=768,
                        depth=12,
                        num_heads=12,
                        **kwargs)
    model = deit._create_deit('tian2022deepeer_deit_b_imagenet_corruptions',
                              pretrained=pretrained,
                              **model_kwargs)
    assert isinstance(model, deit.VisionTransformer)
    model = normalize_timm_model(model)
    return model

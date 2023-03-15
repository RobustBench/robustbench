"""Definition of ConvStem models as in https://arxiv.org/abs/2303.01870."""

import torch
import torch.nn as nn
from collections import OrderedDict
from typing import Tuple
from torch import Tensor
import torch.nn as nn

import timm
from timm.models import create_model
import torch.nn.functional as F
import math

from robustbench.model_zoo.architectures.utils_architectures import normalize_model


IMAGENET_MEAN = [c * 1. for c in (0.485, 0.456, 0.406)]
IMAGENET_STD = [c * 1. for c in (0.229, 0.224, 0.225)] 


class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first. 
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with 
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs 
    with shape (batch_size, channels, height, width).

    From https://github.com/facebookresearch/ConvNeXt/blob/main/models/convnext.py.
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_first"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x
        
        
class ConvBlock(nn.Module):
    expansion = 1
    def __init__(self, siz=48, end_siz=8, fin_dim=384):
        super(ConvBlock, self).__init__()
        self.planes = siz
        fin_dim = self.planes * end_siz if fin_dim != 432 else 432
        # self.bn = nn.BatchNorm2d(planes) if self.normaliz == "bn" else nn.GroupNorm(num_groups=1, num_channels=planes)
        self.stem = nn.Sequential(nn.Conv2d(3, self.planes, kernel_size=3, stride=2, padding=1),
                                  LayerNorm(self.planes, data_format="channels_first"),
                                  nn.GELU(),
                                  nn.Conv2d(self.planes, self.planes*2, kernel_size=3, stride=2, padding=1),
                                  LayerNorm(self.planes*2, data_format="channels_first"),
                                  nn.GELU(),
                                  nn.Conv2d(self.planes*2, self.planes*4, kernel_size=3, stride=2, padding=1),
                                  LayerNorm(self.planes*4, data_format="channels_first"),
                                  nn.GELU(),
                                  nn.Conv2d(self.planes*4, self.planes*8, kernel_size=3, stride=2, padding=1),
                                  LayerNorm(self.planes*8, data_format="channels_first"),
                                  nn.GELU(),
                                  nn.Conv2d(self.planes*8, fin_dim, kernel_size=1, stride=1, padding=0)
                        )
    def forward(self, x):
        out = self.stem(x)
        # out = self.bn(out)
        return out


class ConvBlock3(nn.Module):
    # expansion = 1
    def __init__(self, siz=64):
        super(ConvBlock3, self).__init__()
        self.planes = siz

        self.stem = nn.Sequential(nn.Conv2d(3, self.planes, kernel_size=3, stride=2, padding=1),
                                  LayerNorm(self.planes, data_format="channels_first"),
                                  nn.GELU(),
                                  nn.Conv2d(self.planes, int(self.planes*1.5), kernel_size=3, stride=2, padding=1),
                                  LayerNorm(int(self.planes*1.5), data_format="channels_first"),
                                  nn.GELU(),
                                  nn.Conv2d(int(self.planes*1.5), self.planes*2, kernel_size=3, stride=1, padding=1),
                                  LayerNorm(self.planes*2, data_format="channels_first"),
                                  nn.GELU()
                                  )

    def forward(self, x):
        out = self.stem(x)
        # out = self.bn(out)
        return out


class ConvBlock1(nn.Module):
    def __init__(self, siz=48, end_siz=8, fin_dim=384):
        super(ConvBlock1, self).__init__()
        self.planes = siz

        fin_dim = self.planes*end_siz if fin_dim == None else 432
        self.stem = nn.Sequential(nn.Conv2d(3, self.planes, kernel_size=3, stride=2, padding=1),
                                  LayerNorm(self.planes, data_format="channels_first"),
                                  nn.GELU(),
                                  nn.Conv2d(self.planes, self.planes*2, kernel_size=3, stride=2, padding=1),
                                  LayerNorm(self.planes*2, data_format="channels_first"),
                                  nn.GELU()
                                  )

    def forward(self, x):
        out = self.stem(x)
        # out = self.bn(out)
        return out


def get_convstem_models(modelname, pretrained=False):
    """Initialize models with ConvStem."""

    if modelname == 'convnext_t_cvst':
        model = timm.models.convnext.convnext_tiny(pretrained=pretrained)
        model.stem = ConvBlock1(48, end_siz=8)

    elif modelname == "convnext_s_cvst":
        model = timm.models.convnext.convnext_small(pretrained=pretrained)
        model.stem = ConvBlock1(48, end_siz=8)

    elif modelname == "convnext_b_cvst":
        model_args = dict(depths=[3, 3, 27, 3], dims=[128, 256, 512, 1024])
        model = timm.models.convnext._create_convnext(
            'convnext_base.fb_in1k', pretrained=pretrained, **model_args)
        model.stem = ConvBlock3(64)
        
    elif modelname == "convnext_l_cvst":
        model = timm.models.convnext_large(pretrained=pretrained)
        model.stem = ConvBlock3(96)
        
    elif modelname == 'vit_s_cvst':
        model = create_model('deit_small_patch16_224', pretrained=pretrained)
        model.patch_embed.proj = ConvBlock(48, end_siz=8)
        model = normalize_model(model, IMAGENET_MEAN, IMAGENET_STD)
        
    elif modelname == 'vit_b_cvst':
        model = timm.models.vision_transformer.vit_base_patch16_224(pretrained=pretrained)
        model.patch_embed.proj = ConvBlock(48, end_siz=16, fin_dim=None)

    else:
        raise ValueError(f'Invalid model name: {modelname}.')

    return model


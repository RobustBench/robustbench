from collections import OrderedDict

import timm
from torchvision import models as pt_models

from robustbench.model_zoo.enums import ThreatModel
from robustbench.model_zoo.architectures.utils_architectures import normalize_model
from robustbench.model_zoo.architectures import alexnet, xcit, deit  # needed to register models
from robustbench.model_zoo.architectures.convstem_models import get_convstem_models
from robustbench.model_zoo.architectures.robustarch_wide_resnet import get_model as get_robustarch_model
from robustbench.model_zoo.architectures.comp_model import get_nonlin_mixed_classifier
from robustbench.model_zoo.architectures.sparsified_model import get_sparse_model


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
        ('Debenedetti2022Light_XCiT-S12', {
            'model': (lambda: timm.create_model(
                'debenedetti2020light_xcit_s_imagenet_linf', pretrained=True)),
            'gdrive_id':
            None
        }),
        ('Debenedetti2022Light_XCiT-M12', {
            'model': (lambda: timm.create_model(
                'debenedetti2020light_xcit_m_imagenet_linf', pretrained=True)),
            'gdrive_id':
            None
        }),
        ('Debenedetti2022Light_XCiT-L12', {
            'model': (lambda: timm.create_model(
                'debenedetti2020light_xcit_l_imagenet_linf', pretrained=True)),
            'gdrive_id':
            None
        }),
        ('Singh2023Revisiting_ViT-S-ConvStem', {
            'model': lambda: get_convstem_models('vit_s_cvst'),
            'gdrive_id': '1-1sUYXnj6bDXacIKI3KKqn4rlkmL-ZI2',
            'preprocessing': 'BicubicRes256Crop224'
        }),
        ('Singh2023Revisiting_ViT-B-ConvStem', {
            'model': lambda: get_convstem_models('vit_b_cvst'),
            'gdrive_id': '1-JBbfi_eH3tKMXObvPPHprrZae0RiQGT',
            'preprocessing': 'BicubicRes256Crop224'
        }),
        ('Singh2023Revisiting_ConvNeXt-T-ConvStem', {
            'model': lambda: get_convstem_models('convnext_t_cvst'),
            'gdrive_id': '1-FjtOF6LJ3-bf4VezsmWwncCxYSx-USP',
            'preprocessing': 'BicubicRes256Crop224'
        }),
        ('Singh2023Revisiting_ConvNeXt-S-ConvStem', {
            'model': lambda: get_convstem_models('convnext_s_cvst'),
            'gdrive_id': '1-ZrMYajCCnrtV4oT0wa3qJJoQy1nUSnL',
            'preprocessing': 'BicubicRes256Crop224'
        }),
        ('Singh2023Revisiting_ConvNeXt-B-ConvStem', {
            'model': lambda: get_convstem_models('convnext_b_cvst'),
            'gdrive_id': '1-lE-waaVvfL7lgBrydmZIM9UJimmHnVe',
            'preprocessing': 'BicubicRes256Crop224'
        }),
        ('Singh2023Revisiting_ConvNeXt-L-ConvStem', {
            'model': lambda: get_convstem_models('convnext_l_cvst'),
            'gdrive_id': '10-YOVdM2EQjHemSi9x2H44qKRSOXVQmh',
            'preprocessing': 'BicubicRes256Crop224'
        }),
        ('Liu2023Comprehensive_ConvNeXt-B', {
            'model': lambda: normalize_model(
                timm.create_model('convnext_base', pretrained=False), mu, sigma),
            'gdrive_id': '10-nSm-qUftvfKXHeOAakBQl8rxm-jCbk',
            'preprocessing': 'BicubicRes256Crop224',
        }),
        ('Liu2023Comprehensive_ConvNeXt-L', {
            'model': lambda: normalize_model(
                timm.create_model('convnext_large', pretrained=False), mu, sigma),
            'gdrive_id': '1dIPLNfdQtAnqZrKPyuy3_zDI-FVgJ2FH',
            'preprocessing': 'BicubicRes256Crop224'
        }),
        ('Liu2023Comprehensive_Swin-B', {
            'model': lambda: normalize_model(timm.create_model(
                'swin_base_patch4_window7_224', pretrained=False), mu, sigma),
            'gdrive_id': '1-4mtxQCkThJUVdS3wvQ6NnmMZuySqR3c',
            'preprocessing': 'BicubicRes256Crop224'
        }),
        ('Liu2023Comprehensive_Swin-L', {
            'model': lambda: normalize_model(timm.create_model(
                'swin_large_patch4_window7_224', pretrained=False), mu, sigma),
            'gdrive_id': '1-57sQfcrsDsslfDR18nRD7FnpQmsSBk7',
            'preprocessing': 'BicubicRes256Crop224'
        }),
        ('Peng2023Robust', {
            'model': lambda: get_robustarch_model('ra_wrn101_2'),  # TODO: check device calls.
            'gdrive_id': '1-GpZ9Du83mBTN61Ytx9z_ZQSyIc1kYop',
            'preprocessing': 'Res256Crop224',
        }),
        ('Bai2024MixedNUTS', {
            'model': lambda: get_nonlin_mixed_classifier('imagenet'),  # TODO: check device calls.
            'gdrive_id': [
                '1-2CwsRuMZXr99PeU2Z7-iFjl_86uIR-Y',
                '1-57sQfcrsDsslfDR18nRD7FnpQmsSBk7'],
            'preprocessing': 'BicubicRes256Crop224'
        }),
        ('Chen2024Data_WRN_50_2', {
            'model': lambda: pt_models.resnet50(width_per_group=64 * 2),
            'gdrive_id': '1-PBlZVILAKFQ7mF8srKjdkTKJZAr61Uf',
            'preprocessing': 'Res256Crop224',
        }),
        ('Mo2022When_Swin-B', {
            'model': lambda: normalize_model(timm.create_model(
                'swin_base_patch4_window7_224', pretrained=False,
                ), mu, sigma),
            'gdrive_id': '1-SXi4Z2X6Zo_j8EO4slJcBMXNej8fKUd',
            'preprocessing': 'Res224',
        }),
        ('Mo2022When_ViT-B', {
            'model': lambda: normalize_model(timm.create_model(
                'vit_base_patch16_224', pretrained=False,
                ), mu, sigma),
            'gdrive_id': '1-dUFdvDBflqMsMLjZv3wlPJTm-Jm7net',
            'preprocessing': 'Res224',
        }),
        ('Amini2024MeanSparse_ConvNeXt-L', {
            'model': lambda: get_sparse_model(
                normalize_model(timm.create_model('convnext_large', pretrained=False),
                mu, sigma), dataset='imagenet-Linf'),
            'gdrive_id': '1-LUMPqauSx68bPmZFIuklFoJ6NmBhu7A',
            'preprocessing': 'BicubicRes256Crop224',
        }),
        ('Amini2024MeanSparse_Swin-L', {
            'model': lambda: get_sparse_model('swin-l', dataset='imagenet-Linf'),
            'gdrive_id': '1-KmvrDXd_kcJS-TcNmtHP5NInQ5I4lgS',
            'preprocessing': 'BicubicRes256Crop224',
        }),
        ('RodriguezMunoz2024Characterizing_Swin-B', {
            'model': lambda: normalize_model(timm.create_model(
                'swin_base_patch4_window7_224', pretrained=False), mu, sigma),
            'gdrive_id': '1-BSUjoFXx3PP-TfeE5fjbofO2lLyUf56',  # '1-9h_4PImbQM3XhKBcnqTh4PHxz9rM6vr',
            'preprocessing': 'BicubicRes256Crop224'
        }),
        ('RodriguezMunoz2024Characterizing_Swin-L', {
            'model': lambda: normalize_model(timm.create_model(
                'swin_large_patch4_window7_224', pretrained=False), mu, sigma),
            'gdrive_id': '1-Dc9WhPU2wv4OMskLo1U57n5O8VbpNXv',  # '1-DoJoTiPynr39AFNsEyhOej4rKPL3xqT'
            'preprocessing': 'BicubicRes256Crop224'
        }),
    ])

common_corruptions = OrderedDict(
    [
        ('Erichson2022NoisyMix', {
            'model': lambda: normalize_model(pt_models.resnet50(), mu, sigma),
            'gdrive_id': '1wFsiB5h4Fgv6HqOkI_1jR1SZOB_eNK9r',
            'preprocessing': 'Res256Crop224'
        }),
        ('Erichson2022NoisyMix_new', {
            'model': lambda: normalize_model(pt_models.resnet50(), mu, sigma),
            'gdrive_id': '1Na79fzPZ0Azg01h6kGn1Xu5NoWOElSuG',
            'preprocessing': 'Res256Crop224'
        }),
        ('Geirhos2018_SIN', {
            'model': lambda: normalize_model(pt_models.resnet50(), mu, sigma),
            'gdrive_id': '1hLgeY_rQIaOT4R-t_KyOqPNkczfaedgs',
            'preprocessing': 'Res256Crop224'
        }),
        ('Geirhos2018_SIN_IN', {
            'model': lambda: normalize_model(pt_models.resnet50(), mu, sigma),
            'gdrive_id': '139pWopDnNERObZeLsXUysRcLg6N1iZHK',
            'preprocessing': 'Res256Crop224'
        }),
        ('Geirhos2018_SIN_IN_IN', {
            'model': lambda: normalize_model(pt_models.resnet50(), mu, sigma),
            'gdrive_id': '1xOvyuxpOZ8I5CZOi0EGYG_R6tu3ZaJdO',
            'preprocessing': 'Res256Crop224'
        }),
        ('Hendrycks2020Many', {
            'model': lambda: normalize_model(pt_models.resnet50(), mu, sigma),
            'gdrive_id': '1kylueoLtYtxkpVzoOA1B6tqdbRl2xt9X',
            'preprocessing': 'Res256Crop224'
        }),
        ('Hendrycks2020AugMix', {
            'model': lambda: normalize_model(pt_models.resnet50(), mu, sigma),
            'gdrive_id': '1xRMj1GlO93tLoCMm0e5wEvZwqhIjxhoJ',
            'preprocessing': 'Res256Crop224'
        }),
        ('Salman2020Do_50_2_Linf', {
            'model': lambda: normalize_model(pt_models.wide_resnet50_2(), mu, sigma),
            'gdrive_id': '1OT7xaQYljrTr3vGbM37xK9SPoPJvbSKB',
            'preprocessing': 'Res256Crop224'
        }),
        ('Standard_R50', {
            'model': lambda: normalize_model(pt_models.resnet50(pretrained=True), mu, sigma),
            'gdrive_id': '',
            'preprocessing': 'Res256Crop224'
        }),
        ('AlexNet', {
            'model': (lambda: timm.create_model(
                'alexnet', pretrained=True)),
            'gdrive_id': None,
            'preprocessing': 'Res256Crop224'
        }),
        ('Tian2022Deeper_DeiT-S', {
            'model': (lambda: timm.create_model(
                'tian2022deeper_deit_s_imagenet_corruptions', pretrained=True)),
            'gdrive_id': None,
            'preprocessing': 'Res256Crop224'
        }),
        ('Tian2022Deeper_DeiT-B', {
            'model': (lambda: timm.create_model(
                'tian2022deeper_deit_b_imagenet_corruptions', pretrained=True)),
            'gdrive_id': None,
            'preprocessing': 'Res256Crop224'
        }),
    ])

imagenet_models = OrderedDict([(ThreatModel.Linf, linf),
                               (ThreatModel.corruptions, common_corruptions)])



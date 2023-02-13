"""Implementation of RobustResNet from https://arxiv.org/abs/2212.11005."""

import math
import torch
import torch.nn as nn


avaliable_activations = {"ReLU": nn.ReLU, "SiLU": nn.SiLU,}
avaliable_normalizations = {"BatchNorm": nn.BatchNorm2d,}


class PreActBasicBlock(nn.Module):
    def __init__(self, in_planes, planes, stride, kernel_size=3, activation='ReLU',
                 normalization='BatchNorm', **kwargs):
        super(PreActBasicBlock, self).__init__()
        self.act = avaliable_activations[activation](inplace=True)
        self.bn1 = avaliable_normalizations[normalization](in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=kernel_size, stride=stride, padding=kernel_size // 2,
                               bias=False)
        self.bn2 = avaliable_normalizations[normalization](planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=kernel_size, stride=1, padding=kernel_size // 2, bias=False)

        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, padding=0, bias=False)
            )

    def forward(self, x):
        out = self.act(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(self.act(self.bn2(out)))
        return out + shortcut


class NetworkBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, out_planes, stride, kernel_size=3, block_type='basic_block',
                 cardinality=8, base_width=64, scales=4, activation='ReLU', normalization='BatchNorm',
                 se_reduction=16, ):
        super(NetworkBlock, self).__init__()
        self.block_type = block_type
        if block_type == 'basic_block':
            block = PreActBasicBlock
        elif block_type == 'robust_res_block':
            block = RobustResBlock
        else:
            raise ('Unknown block: %s' % block_type)

        self.layer = self._make_layer(
            block, in_planes, out_planes, nb_layers, stride, kernel_size, activation, normalization,
            cardinality=cardinality, base_width=base_width, scales=scales, se_reduction=se_reduction)

    def _make_layer(self, block, in_planes, out_planes, nb_layers, stride, kernel_size, activation,
                    normalization, cardinality=8, base_width=64, scales=4, se_reduction=16, ):
        layers = []
        for i in range(int(nb_layers)):
            if i == 0:
                in_planes = in_planes
            else:
                if self.block_type == 'robust_res_block':
                    in_planes = out_planes * 4
                else:
                    in_planes = out_planes

            layers.append(block(in_planes, out_planes, i == 0 and stride or 1,
                                kernel_size=kernel_size,
                                activation=activation, normalization=normalization,
                                cardinality=cardinality, base_width=base_width,
                                scales=scales, se_reduction=se_reduction)
                          )
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)


class PreActResNet(nn.Module):
    def __init__(self, num_classes=10, channel_configs=(16, 160, 320, 640),
                 depth_configs=(5, 5, 5), drop_rate_config=(0.0, 0.0, 0.0),
                 stride_config=(1, 2, 2), zero_init_residual=False, stem_stride=1,
                 kernel_size_configs=(3, 3, 3),
                 block_types=('basic_block', 'basic_block', 'basic_block'),
                 activations=('ReLU', 'ReLU', 'ReLU'),
                 normalizations=('BatchNorm', 'BatchNorm', 'BatchNorm'),
                 use_init=True, cardinality=8, base_width=64, scales=4,
                 se_reduction=16, pre_process=False):
        super(PreActResNet, self).__init__()
        assert len(channel_configs) - 1 == len(depth_configs) == len(stride_config)
        self.channel_configs = channel_configs
        self.depth_configs = depth_configs
        self.stride_config = stride_config
        self.get_feature = False
        self.get_stem_out = False
        self.block_types = block_types

        self.pre_process = pre_process
        # if True, add data normalization, this is only used for advanced training on CIFAR-10
        if pre_process:
            self.register_buffer(
                'mu', torch.tensor((0.4914, 0.4822, 0.4465)).view(3, 1, 1))
            self.register_buffer(
                'sigma', torch.tensor((0.2471, 0.2435, 0.2616)).view(3, 1, 1))
        

        self.stem_conv = nn.Conv2d(
            3, channel_configs[0], kernel_size=3, stride=stem_stride, padding=1,
            bias=False)
        self.blocks = nn.ModuleList([])

        out_planes = channel_configs[0]
        for i, stride in enumerate(stride_config):
            self.blocks.append(NetworkBlock(nb_layers=depth_configs[i],
                                            in_planes=out_planes,
                                            out_planes=channel_configs[i + 1],
                                            stride=stride,
                                            kernel_size=kernel_size_configs[i],
                                            block_type=block_types[i],
                                            activation=activations[i],
                                            normalization=normalizations[i],
                                            cardinality=cardinality,
                                            base_width=base_width,
                                            scales=scales,
                                            se_reduction=se_reduction,
                                            ))
            if block_types[i] == 'robust_res_block':
                out_planes = channel_configs[i + 1] * 4
            else:
                out_planes = channel_configs[i + 1]

        # global average pooling and classifier
        self.norm1 = avaliable_normalizations[normalizations[-1]](out_planes)
        self.act1 = avaliable_activations[activations[-1]](inplace=True)
        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(out_planes, num_classes)
        self.fc_size = out_planes
        if use_init:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                    m.weight.data.normal_(0, math.sqrt(2. / n))
                elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.GroupNorm):
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()
                elif isinstance(m, nn.Linear):
                    if m.bias is not None:
                        m.bias.data.zero_()

    def forward(self, x):

        if self.pre_process:
            x = (x - self.mu) / self.sigma

        out = self.stem_conv(x)
        for i, block in enumerate(self.blocks):
            out = block(out)
        out = self.act1(self.norm1(out))
        out = self.global_pooling(out)
        out = out.view(-1, self.fc_size)
        out = self.fc(out)
        return out


def get_model(modelname, num_classes=10):
    """Initialize model."""

    if modelname == 'WRN-A4':
        stride_config = [1, 2, 2]
        activations = ('SiLU', 'SiLU', 'SiLU')
        normalizations = ('BatchNorm', 'BatchNorm', 'BatchNorm')
        depth, width_mult = [27, 28, 13], [10, 14, 6]
        block_types = ['basic_block', 'basic_block', 'basic_block']
        scales, base_width, cardinality, se_reduction = None, None, None, None
        channels = [16, 16 * width_mult[0], 32 * width_mult[1], 64 * width_mult[2]]
        model = PreActResNet(
            num_classes=num_classes,
            channel_configs=channels,
            depth_configs=depth,
            stride_config=stride_config,
            stem_stride=1,
            block_types=block_types,
            activations=activations,
            normalizations=normalizations,
            use_init=False,
            cardinality=cardinality,
            base_width=base_width,
            scales=scales,
            se_reduction=se_reduction,
            pre_process=True,
        )

    else:
        raise ValueError(f'Unknown model version: {modelname}.')

    return model

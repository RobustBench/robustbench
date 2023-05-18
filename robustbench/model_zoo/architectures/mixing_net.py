import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class PreActBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False)
            )

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out)
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out += shortcut
        return out


class MixingNetV3(nn.Module):
    def __init__(self, forward_settings, nmodels=2):
        super().__init__()
        #print("Initializing MixingNet V3.")
        self.nmodels = nmodels

        self.ind_planes = [None for _ in range(self.nmodels)]
        for ind, in_plane in enumerate(forward_settings["in_planes"]):
            if in_plane == 64:  # Number of channels of each layer
                self.ind_planes[ind] = (64, 128, 256, 512)
            elif in_plane == 160:
                self.ind_planes[ind] = (160, 320, 512, 768)
            elif in_plane == 256:
                self.ind_planes[ind] = (256, 256, 384, 512)
            elif in_plane == 512:
                self.ind_planes[ind] = (512, 256, 384, 512)
            else:
                raise ValueError("Unknown in_plane.")
        self.planes = np.array(self.ind_planes).sum(axis=0)

        self.in_planes = self.planes[0]
        self.layers = nn.ModuleList([self._make_layer(pl, stride=2) for pl in self.planes[1:]])

        self.global_avg_pool = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(self.planes[-1], 1, bias=False)

    def _make_layer(self, planes, stride, num_blocks=2):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(PreActBlock(self.in_planes, planes, stride))
            self.in_planes = planes * PreActBlock.expansion
        return nn.Sequential(*layers)

    def forward(self, feats1, feats2):
        feats1, feats2 = torch.cat(feats1, dim=1), torch.cat(feats2, dim=1)
        feats1 = self.layers[0](feats1)

        assert torch.all(feats1.shape == feats2.shape)
        feats = feats1 + feats2
        for l in range(1, len(self.layers)):
            feats = self.layers[l](feats)

        feats = self.global_avg_pool(feats).reshape(-1, self.planes[-1])
        feats = self.linear(feats)
        return feats


class MixingNetV4(MixingNetV3):
    def __init__(self, forward_settings, nmodels=2):
        super().__init__(forward_settings, nmodels)
        #print("Initializing MixingNet V4.")

        # Add 1x1 conv to reduce the number of channels
        in_1x1 = (self.ind_planes[0][0] + self.ind_planes[1][0]) * 2
        out_1x1 = self.ind_planes[0][1] + self.ind_planes[1][1]
        self.conv1x1 = nn.Conv2d(in_1x1, out_1x1, kernel_size=1, stride=1, bias=False)

    def forward(self, feats1, feats2):
        feats1, feats2 = torch.cat(feats1, dim=1), torch.cat(feats2, dim=1)
        feats = self.layers[0](feats1) + self.conv1x1(feats2)
        for l in range(1, len(self.layers)):
            feats = self.layers[l](feats)

        feats = self.global_avg_pool(feats).reshape(-1, self.planes[-1])
        feats = self.linear(feats)
        return feats

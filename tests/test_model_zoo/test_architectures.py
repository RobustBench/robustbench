from typing import Union
from unittest import TestCase

from torch import nn

from robustbench import load_cifar10
from robustbench.model_zoo.architectures.dm_wide_resnet import DMWideResNet
from robustbench.model_zoo.architectures.resnet import PreActBlock, PreActResNet, \
    PreActResNet18, \
    ResNet18
from robustbench.model_zoo.architectures.resnext import CifarResNeXt, ResNeXtBottleneck
from robustbench.model_zoo.architectures.utils import LipschitzModel
from robustbench.model_zoo.architectures.wide_resnet import WideResNet


class ArchitecturesTester(TestCase):
    x, _ = load_cifar10(10)

    def _test_arch(self, net: Union[LipschitzModel, nn.Module]):
        # Test that the full sequence is actually equivalent to the full model
        exp_result = net(self.x)
        seq_net = nn.Sequential(*net.get_lipschitz_layers())
        result = seq_net(self.x)
        self.assertEqual(exp_result.shape, result.shape)
        self.assertTrue((exp_result == exp_result).all().item())

    def test_dm_wide_resnet(self):
        self._test_arch(DMWideResNet())

    def test_wide_resnet(self):
        self._test_arch(WideResNet())

    def test_preact_resnet(self):
        self._test_arch(PreActResNet18())

    def test_preact_resnet_bn_before_fc(self):
        self._test_arch(PreActResNet(PreActBlock, [2, 2, 2, 2], bn_before_fc=True))

    def test_resnet(self):
        self._test_arch(ResNet18())

    def test_cifar_resnext(self):
        self._test_arch(
            CifarResNeXt(ResNeXtBottleneck, depth=29, num_classes=10, cardinality=4, base_width=32))

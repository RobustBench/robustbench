from unittest import TestCase

import torch

from robustbench.eval.lipschitz import box, compute_lipschitz
from tests.utils import DummyModel


class LipschitzTester(TestCase):
    def test_box(self):
        eps = 0.2
        x = torch.Tensor([0, 0.5, 0.89])
        x_prime = torch.Tensor([0.3, 0.6, 1.2])
        boxed_x_prime = box(x_prime, x, eps)
        expected = torch.Tensor([0.2, 0.6, 1.0])
        self.assertTrue((boxed_x_prime.numpy() == expected.numpy()).all())

    def test_benchmark_train(self):
        in_features = 1
        model = DummyModel(in_shape=in_features, out_shape=1)
        slope = 5.0

        for parameter in model.parameters():
            parameter.data = slope * torch.ones(parameter.data.shape)

        eps = 8 / 255
        x = torch.randn(25, in_features)
        lips = compute_lipschitz(model, x, eps, eps / 5, 50)

        self.assertAlmostEqual(lips, slope, places=2)

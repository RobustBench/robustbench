import random
from unittest import TestCase

import torch
from torch.utils.data import DataLoader, TensorDataset

from robustbench.eval.lipschitz import benchmark_lipschitz, box, compute_lipschitz, \
    compute_lipschitz_batch
from tests.utils import DummyModel


class LipschitzTester(TestCase):
    def test_box(self):
        eps = 0.2
        x = torch.Tensor([0, 0.5, 0.89])
        x_prime = torch.Tensor([0.3, 0.6, 1.2])
        boxed_x_prime = box(x_prime, x, eps)
        expected = torch.Tensor([0.2, 0.6, 1.0])
        self.assertTrue((boxed_x_prime.numpy() == expected.numpy()).all())

    def _test_compute_lipschitz_batch(self, normalize, assertion):
        model = DummyModel(in_shape=1, out_shape=1, slope=random.random())

        eps = 8 / 255
        x = torch.randn(200, model.in_shape)
        lips = compute_lipschitz_batch(model,
                                       x,
                                       eps,
                                       eps / 5,
                                       50,
                                       l2_normalize=normalize,
                                       p=float("inf"))

        assertion(model, x, lips)

    def test_compute_lipschitz_batch(self):
        self._test_compute_lipschitz_batch(
            False,
            lambda f, _, lips: self.assertAlmostEqual(lips, f.slope, places=2))

    def test_compute_lipschitz_batch_norm(self):
        self._test_compute_lipschitz_batch(
            True, lambda f, _, lips: self.assertGreaterEqual(lips, 0))

    def test_compute_lipschitz(self):
        model = DummyModel(in_shape=1, out_shape=1, slope=random.random())

        eps = 8 / 255
        x = torch.randn(200, model.in_shape)
        y = torch.randn(200, model.out_shape)
        dl = DataLoader(TensorDataset(x, y), batch_size=50)
        lips = compute_lipschitz(model,
                                 dl,
                                 eps,
                                 eps / 5,
                                 50,
                                 l2_normalize=False)

        self.assertAlmostEqual(lips, model.slope, places=2)

    def test_benchmark_lipschitz(self):
        model = DummyModel()
        lips = benchmark_lipschitz(model.eval(),
                                   16,
                                   "cifar10",
                                   l2_normalize=False)
        self.assertGreaterEqual(lips[0], 0)

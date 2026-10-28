"""Regression tests for benchmark(aa_state_path=...) (issue #217).

`EvaluationState.from_disk` rebuilds `robust_flags` as a torch.bool
tensor. On current PyTorch, calling `.mean()` on a bool tensor raises
`RuntimeError: mean(): could not infer output dtype`, which crashed the
aa_state_path branch of `benchmark()`.

The test exercises the real robustbench code path (benchmark with a
state file produced by AutoAttack's own to_disk()) while staying fast
and CPU-only: the dataset loader and
`AutoAttack.run_standard_evaluation` are replaced with fakes. The fake
attack writes the same completed state file the real attack writes
(through `EvaluationState`'s own save path), then returns the clean
inputs.
"""

import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch
from autoattack import AutoAttack
from autoattack.state import EvaluationState

import robustbench.eval as rb_eval
from robustbench import benchmark
from tests.utils import DummyModel


def _fake_load_clean_dataset(dataset, n_examples, data_dir, transforms=None):
    x = torch.rand(2, 3, 32, 32)
    y = torch.tensor([0, 1])
    return x, y


def _fake_run_standard_evaluation(self, x_orig, y_orig, bs=1,
                                  state_path=None, **kwargs):
    """Write the same completed state file AutoAttack writes, then return."""
    n = x_orig.shape[0]
    flags = torch.ones(n, dtype=torch.bool)
    flags[-1] = False  # last example is misclassified under attack
    state = EvaluationState(_attacks_to_run=set(self.attacks_to_run),
                            path=state_path)
    state.robust_flags = flags  # the setter saves the state to disk
    return x_orig


class BenchmarkAAStatePathTester(unittest.TestCase):
    def test_benchmark_with_aa_state_path(self):
        model = DummyModel()
        model.eval()
        with tempfile.TemporaryDirectory() as tmp:
            state_path = Path(tmp) / "aa_state.json"
            with mock.patch.object(rb_eval, "load_clean_dataset",
                                   _fake_load_clean_dataset), \
                 mock.patch.object(AutoAttack, "run_standard_evaluation",
                                   _fake_run_standard_evaluation):
                _, adv_acc = benchmark(model,
                                       dataset="cifar10",
                                       threat_model="Linf",
                                       eps=8 / 255,
                                       n_examples=2,
                                       batch_size=2,
                                       device=torch.device("cpu"),
                                       aa_state_path=state_path)
        # One of the two examples is non-robust in the saved state.
        self.assertAlmostEqual(adv_acc, 0.5)

    def test_bool_flags_mean_raises_without_cast(self):
        # Documents the original failure mode this guards against: the
        # accuracy must be computed with a float cast.
        flags = torch.tensor([True, False, True])
        with self.assertRaises(RuntimeError):
            flags.mean()
        self.assertAlmostEqual(flags.float().mean().item(), 2 / 3)


if __name__ == "__main__":
    unittest.main()

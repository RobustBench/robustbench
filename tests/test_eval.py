from unittest import TestCase

from robustbench import benchmark
from robustbench.model_zoo.enums import ThreatModel
from tests.utils import DummyModel


class CleanAccTester(TestCase):
    def test_benchmark_train(self):
        model = DummyModel()
        model.train()
        with self.assertWarns(Warning):
            benchmark(model,
                      n_examples=1,
                      threat_model=ThreatModel.Linf,
                      eps=8/255, batch_size=100)


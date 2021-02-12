from collections import OrderedDict
from typing import Any, Dict

from robustbench.model_zoo.cifar10 import cifar_10_models
from robustbench.model_zoo.enums import BenchmarkDataset, ThreatModel

ModelsDict = Dict[str, Dict[str, Any]]
ThreatModelsDict = Dict[ThreatModel, ModelsDict]
BenchmarkDict = Dict[BenchmarkDataset, ThreatModelsDict]

model_dicts: BenchmarkDict = OrderedDict([
    (BenchmarkDataset.cifar_10, cifar_10_models)
])

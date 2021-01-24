from collections import OrderedDict
from typing import OrderedDict as OrderedDictType

from torch import nn

from robustbench.model_zoo.cifar10 import cifar_10_models
from robustbench.model_zoo.enums import BenchmarkDataset, ThreatModel

ModelsDict = OrderedDictType[str, nn.Module]
ThreatModelsDict = OrderedDictType[ThreatModel, ModelsDict]
BenchmarkDict = OrderedDictType[BenchmarkDataset, ThreatModelsDict]

model_dicts: BenchmarkDict = OrderedDict([
    (BenchmarkDataset.cifar_10, cifar_10_models)
])

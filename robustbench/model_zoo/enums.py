from enum import Enum


class BenchmarkDataset(Enum):
    cifar_10 = 'cifar10'
    cifar_100 = 'cifar100'
    imagenet = 'imagenet'


class ThreatModel(Enum):
    Linf = "Linf"
    L2 = "L2"
    corruptions = "corruptions"

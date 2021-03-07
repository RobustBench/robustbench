from enum import Enum


class BenchmarkDataset(Enum):
    cifar_10 = 'cifar10'
    cifar_100 = 'cifar100'
    image_net = 'imagenet'


class ThreatModel(Enum):
    L2 = "L2"
    Linf = "Linf"
    corruptions = "corruptions"

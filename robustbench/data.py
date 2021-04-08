import os
from pathlib import Path
from typing import Callable, Dict, Optional, Sequence, Set, Tuple

import numpy as np
import torch
import torch.utils.data as data
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import Dataset

from robustbench.model_zoo.enums import BenchmarkDataset
from robustbench.zenodo_download import DownloadError, zenodo_download


def _load_dataset(
        dataset: Dataset,
        n_examples: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    batch_size = 100
    test_loader = data.DataLoader(dataset,
                                  batch_size=batch_size,
                                  shuffle=False,
                                  num_workers=0)

    x_test, y_test = [], []
    for i, (x, y) in enumerate(test_loader):
        x_test.append(x)
        y_test.append(y)
        if n_examples is not None and batch_size * i >= n_examples:
            break
    x_test_tensor = torch.cat(x_test)
    y_test_tensor = torch.cat(y_test)

    if n_examples is not None:
        x_test_tensor = x_test_tensor[:n_examples]
        y_test_tensor = y_test_tensor[:n_examples]

    return x_test_tensor, y_test_tensor


def load_cifar10(
        n_examples: Optional[int] = None,
        data_dir: str = './data') -> Tuple[torch.Tensor, torch.Tensor]:
    transform_chain = transforms.Compose([transforms.ToTensor()])
    dataset = datasets.CIFAR10(root=data_dir,
                               train=False,
                               transform=transform_chain,
                               download=True)
    return _load_dataset(dataset, n_examples)


def load_cifar100(
        n_examples: Optional[int] = None,
        data_dir: str = './data') -> Tuple[torch.Tensor, torch.Tensor]:
    transform_chain = transforms.Compose([transforms.ToTensor()])
    dataset = datasets.CIFAR100(root=data_dir,
                                train=False,
                                transform=transform_chain,
                                download=True)
    return _load_dataset(dataset, n_examples)


CleanDatasetLoader = Callable[[Optional[int], str], Tuple[torch.Tensor,
                                                          torch.Tensor]]
_clean_dataset_loaders: Dict[BenchmarkDataset, CleanDatasetLoader] = {
    BenchmarkDataset.cifar_10: load_cifar10,
    BenchmarkDataset.cifar_100: load_cifar100
}


def load_clean_dataset(dataset: BenchmarkDataset, n_examples: Optional[int],
                       data_dir: str) -> Tuple[torch.Tensor, torch.Tensor]:
    return _clean_dataset_loaders[dataset](n_examples, data_dir)


CORRUPTIONS = ("shot_noise", "motion_blur", "snow", "pixelate",
               "gaussian_noise", "defocus_blur", "brightness", "fog",
               "zoom_blur", "frost", "glass_blur", "impulse_noise", "contrast",
               "jpeg_compression", "elastic_transform")

CIFAR_100_CORRUPTIONS = CORRUPTIONS + ("spatter", "speckle_noise",
                                       "gaussian_blur", "saturate")

ZENODO_CORRUPTIONS_LINKS: Dict[BenchmarkDataset, Tuple[str, Set[str]]] = {
    BenchmarkDataset.cifar_10: ("2535967", {"CIFAR-10-C.tar"}),
    BenchmarkDataset.cifar_100: ("3555552", {"CIFAR-100-C.tar"})
}

CORRUPTIONS_DIR_NAMES: Dict[BenchmarkDataset, str] = {
    BenchmarkDataset.cifar_10: "CIFAR-10-C",
    BenchmarkDataset.cifar_100: "CIFAR-100-C"
}

DATASET_CORRUPTIONS: Dict[BenchmarkDataset, Tuple[str, ...]] = {
    BenchmarkDataset.cifar_10: CORRUPTIONS,
    BenchmarkDataset.cifar_100: CIFAR_100_CORRUPTIONS
}


def load_cifar10c(
    n_examples: int,
    severity: int = 5,
    data_dir: str = './data',
    shuffle: bool = False,
    corruptions: Sequence[str] = CORRUPTIONS
) -> Tuple[torch.Tensor, torch.Tensor]:
    return load_corruptions_dataset(BenchmarkDataset.cifar_10, n_examples,
                                    severity, data_dir, corruptions, shuffle)


def load_cifar100c(
    n_examples: int,
    severity: int = 5,
    data_dir: str = './data',
    shuffle: bool = False,
    corruptions: Sequence[str] = CIFAR_100_CORRUPTIONS
) -> Tuple[torch.Tensor, torch.Tensor]:
    return load_corruptions_dataset(BenchmarkDataset.cifar_100, n_examples,
                                    severity, data_dir, corruptions, shuffle)


CorruptDatasetLoader = Callable[[int, int, str, bool, Sequence[str]],
                                Tuple[torch.Tensor, torch.Tensor]]
_corruption_dataset_loaders: Dict[BenchmarkDataset, CorruptDatasetLoader] = {
    BenchmarkDataset.cifar_10: load_cifar10c
}


def load_corruptions_dataset(
        dataset: BenchmarkDataset,
        n_examples: int,
        severity: int,
        data_dir: str,
        corruptions: Sequence[str] = CORRUPTIONS,
        shuffle: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
    assert 1 <= severity <= 5
    n_total_cifar = 10000

    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    data_dir = Path(data_dir)
    data_root_dir = data_dir / CORRUPTIONS_DIR_NAMES[dataset]

    if not data_root_dir.exists():
        zenodo_download(*ZENODO_CORRUPTIONS_LINKS[dataset], save_dir=data_dir)

    # Download labels
    labels_path = data_root_dir / 'labels.npy'
    if not os.path.isfile(labels_path):
        raise DownloadError("Labels are missing, try to re-download them.")
    labels = np.load(labels_path)

    x_test_list, y_test_list = [], []
    n_pert = len(corruptions)
    for corruption in corruptions:
        corruption_file_path = data_root_dir / (corruption + '.npy')
        if not corruption_file_path.is_file():
            raise DownloadError(
                f"{corruption} file is missing, try to re-download it.")

        images_all = np.load(corruption_file_path)
        images = images_all[(severity - 1) * n_total_cifar:severity *
                            n_total_cifar]
        n_img = int(np.ceil(n_examples / n_pert))
        x_test_list.append(images[:n_img])
        # Duplicate the same labels potentially multiple times
        y_test_list.append(labels[:n_img])

    x_test, y_test = np.concatenate(x_test_list), np.concatenate(y_test_list)
    if shuffle:
        rand_idx = np.random.permutation(np.arange(len(x_test)))
        x_test, y_test = x_test[rand_idx], y_test[rand_idx]

    # Make it in the PyTorch format
    x_test = np.transpose(x_test, (0, 3, 1, 2))
    # Make it compatible with our models
    x_test = x_test.astype(np.float32) / 255
    # Make sure that we get exactly n_examples but not a few samples more
    x_test = torch.tensor(x_test)[:n_examples]
    y_test = torch.tensor(y_test)[:n_examples]

    return x_test, y_test

import dataclasses
import os
from typing import Callable, Dict, Sequence, Tuple

import numpy as np
import torch
import torch.utils.data as data
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import Dataset

from robustbench.model_zoo.enums import BenchmarkDataset
from robustbench.utils import download_gdrive


def _load_dataset(dataset: Dataset,
                  n_examples: int) -> Tuple[torch.Tensor, torch.Tensor]:
    batch_size = 100
    test_loader = data.DataLoader(dataset,
                                  batch_size=batch_size,
                                  shuffle=False,
                                  num_workers=0)

    x_test, y_test = [], []
    for i, (x, y) in enumerate(test_loader):
        x_test.append(x)
        y_test.append(y)
        if batch_size * i >= n_examples:
            break

    x_test_tensor = torch.cat(x_test)[:n_examples]
    y_test_tensor = torch.cat(y_test)[:n_examples]

    return x_test_tensor, y_test_tensor


def load_cifar10(
        n_examples: int,
        data_dir: str = './data') -> Tuple[torch.Tensor, torch.Tensor]:
    transform_chain = transforms.Compose([transforms.ToTensor()])
    dataset = datasets.CIFAR10(root=data_dir,
                               train=False,
                               transform=transform_chain,
                               download=True)
    return _load_dataset(dataset, n_examples)


def load_cifar100(
        n_examples: int,
        data_dir: str = './data') -> Tuple[torch.Tensor, torch.Tensor]:
    transform_chain = transforms.Compose([transforms.ToTensor()])
    dataset = datasets.CIFAR100(root=data_dir,
                                train=False,
                                transform=transform_chain,
                                download=True)
    return _load_dataset(dataset, n_examples)


CleanDatasetLoader = Callable[[int, str], Tuple[torch.Tensor, torch.Tensor]]
_clean_dataset_loaders: Dict[BenchmarkDataset, CleanDatasetLoader] = {
    BenchmarkDataset.cifar_10: load_cifar10,
    BenchmarkDataset.cifar_100: load_cifar100
}


def load_clean_dataset(dataset: BenchmarkDataset, n_examples: int,
                       data_dir: str) -> Tuple[torch.Tensor, torch.Tensor]:
    return _clean_dataset_loaders[dataset](n_examples, data_dir)


@dataclasses.dataclass
class _CorruptionsGDriveIDs:
    """Google Drive IDs of the possible corruptions"""
    shot_noise: str
    motion_blur: str
    snow: str
    pixelate: str
    gaussian_noise: str
    defocus_blur: str
    brightness: str
    fog: str
    zoom_blur: str
    frost: str
    glass_blur: str
    impulse_noise: str
    contrast: str
    jpeg_compression: str
    elastic_transform: str


CORRUPTIONS = tuple(field.name
                    for field in dataclasses.fields(_CorruptionsGDriveIDs))


def _load_corruptions_dataset(
        n_examples: int, severity: int, data_dir: str, shuffle: bool,
        corruptions: Sequence[str], gdrive_ids: _CorruptionsGDriveIDs,
        labels_gdrive_id: str) -> Tuple[torch.Tensor, torch.Tensor]:
    assert 1 <= severity <= 5
    n_total_cifar = 10000

    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    # Download labels
    labels_path = data_dir + '/labels.npy'
    if not os.path.isfile(labels_path):
        download_gdrive(labels_gdrive_id, labels_path)
    labels = np.load(labels_path)

    x_test_list, y_test_list = [], []
    n_pert = len(corruptions)
    for corruption in corruptions:
        corruption_file_path = data_dir + '/' + corruption + '.npy'
        if not os.path.isfile(corruption_file_path):
            download_gdrive(
                dataclasses.asdict(gdrive_ids)[corruption],
                corruption_file_path)
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


def load_cifar10c(
    n_examples: int,
    severity: int = 5,
    data_dir: str = './data',
    shuffle: bool = False,
    corruptions: Sequence[str] = CORRUPTIONS
) -> Tuple[torch.Tensor, torch.Tensor]:
    labels_gdrive_id = '1wW8vnLfPXVJyElQBGmCx1bfUz7QKALxp'
    gdrive_ids = _CorruptionsGDriveIDs(
        pixelate='1QxOQZ9fbDiO__PX5bz3npnHEl58nviYi',
        impulse_noise='1F1XB95bOrI5vdE6IqKagoDCGRwgz02Af',
        contrast='1AKzDIt6W7PZUtySB7tuW63RiT0qCmFIn',
        motion_blur='1KaDN9nkbiCcrnJ9gGtf1HXk1gNiNFHJs',
        gaussian_noise='1AKcle1BPbYp-KExKoAxByvpeKBLJOgjI',
        snow='1C0s9ZoIQa9jpK2apgzAYVA6kVnsbyiVo',
        brightness='1oTw7NLMx0USafsEFo8YnrBd2q9yOyih8',
        frost='1A2RFHlCPvRBRiQoRwp5s04qbf2OnhRun',
        elastic_transform='18ohQA9EQ-nuTuPARzvAb_GnxcdwsUsB-',
        defocus_blur='1R9gBaM9Fshp_rj9ZHzJ5-r-42JRcKFl4',
        shot_noise='1Ka58iub7-hIvW9e5FPsljym3-wSPHlXM',
        glass_blur='19sobK_CKJeqMiwJipAk-u_eHx-sh9xOg',
        zoom_blur='148Lb2f4VbmMpcNYCTskLttC9YKsGlZnk',
        jpeg_compression='13XqvkSnRcfUmSvxHr0Acex3itjAq0T95',
        fog='1NSrbUvrWofmRD1LTmg-RmeaZX2ZC-b6U')

    dataset_dir = os.path.join(data_dir, "cifar10c")

    return _load_corruptions_dataset(n_examples, severity, dataset_dir, shuffle,
                                     corruptions, gdrive_ids, labels_gdrive_id)


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
    corruptions: Sequence[str] = CORRUPTIONS
) -> Tuple[torch.Tensor, torch.Tensor]:
    return _corruption_dataset_loaders[dataset](n_examples, severity, data_dir,
                                                False, corruptions)

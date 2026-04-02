import robustbench
from torchvision.datasets.vision import VisionDataset

import torch
import torch.utils.data as data
import torchvision.transforms as transforms

from PIL import Image

import os
import os.path
import sys

from robustbench import data
from robustbench import loaders

data_dir = '~/imagenet/val'

imagenet = loaders.CustomDatasetFolder(data_dir, robustbench.loaders.default_loader, transform=
     transforms.Compose([
         transforms.Resize(256),
         transforms.CenterCrop(224),
         transforms.ToTensor()]))

torch.manual_seed(0)

test_loader = data.data.DataLoader(imagenet,
                              batch_size=50,
                              shuffle=True,
                              num_workers=3)

x, y, path = next(iter(test_loader))

with open('path_imgs_2.txt', 'w') as f:
    f.write('\n'.join(path))
    f.flush()

import os
import numpy as np
import torch
import torchvision.datasets as datasets
import torch.utils.data as data
import torchvision.transforms as transforms
from advbench.utils import download_gdrive


def load_cifar10(n_examples, data_dir='./data'):
    batch_size = 100
    transform_chain = transforms.Compose([transforms.ToTensor()])
    item = datasets.CIFAR10(root=data_dir, train=False, transform=transform_chain, download=True)
    test_loader = data.DataLoader(item, batch_size=batch_size, shuffle=False, num_workers=0)

    x_test, y_test = [], []
    for i, (x, y) in enumerate(test_loader):
        x_test.append(x)
        y_test.append(y)
        if batch_size * i >= n_examples:
            break

    x_test = torch.cat(x_test)[:n_examples]
    y_test = torch.cat(y_test)[:n_examples]

    return x_test.cuda(), y_test.cuda()


def load_cifar10c(n_examples, severity=5, data_dir='./data', shuffle=False,
                  perturbations=('shot_noise', 'motion_blur', 'snow', 'pixelate', 'gaussian_noise', 'defocus_blur',
                                 'brightness', 'fog', 'zoom_blur', 'frost', 'glass_blur', 'impulse_noise', 'contrast',
                                 'jpeg_compression', 'elastic_transform')):
    assert 1 <= severity <= 5
    n_total_cifar = 10000
    labels_gdrive_id = '1wW8vnLfPXVJyElQBGmCx1bfUz7QKALxp'
    dict_gdrive_ids = {
        'pixelate': '1QxOQZ9fbDiO__PX5bz3npnHEl58nviYi',
        'impulse_noise': '1F1XB95bOrI5vdE6IqKagoDCGRwgz02Af',
        'contrast': '1AKzDIt6W7PZUtySB7tuW63RiT0qCmFIn',
        'motion_blur': '1KaDN9nkbiCcrnJ9gGtf1HXk1gNiNFHJs',
        'gaussian_noise': '1AKcle1BPbYp-KExKoAxByvpeKBLJOgjI',
        'snow': '1C0s9ZoIQa9jpK2apgzAYVA6kVnsbyiVo',
        'brightness': '1oTw7NLMx0USafsEFo8YnrBd2q9yOyih8',
        'frost': '1A2RFHlCPvRBRiQoRwp5s04qbf2OnhRun',
        'elastic_transform': '18ohQA9EQ-nuTuPARzvAb_GnxcdwsUsB-',
        'defocus_blur': '1R9gBaM9Fshp_rj9ZHzJ5-r-42JRcKFl4',
        'shot_noise': '1Ka58iub7-hIvW9e5FPsljym3-wSPHlXM',
        'glass_blur': '19sobK_CKJeqMiwJipAk-u_eHx-sh9xOg',
        'zoom_blur': '148Lb2f4VbmMpcNYCTskLttC9YKsGlZnk',
        'jpeg_compression': '13XqvkSnRcfUmSvxHr0Acex3itjAq0T95',
        'fog': '1NSrbUvrWofmRD1LTmg-RmeaZX2ZC-b6U',
    }

    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    # Download labels
    labels_path = data_dir + '/labels.npy'
    if not os.path.isfile(labels_path):
        download_gdrive(labels_gdrive_id, labels_path)
    labels = np.load(labels_path)

    x_test_list, y_test_list = [], []
    n_pert = len(perturbations)
    for corruption in perturbations:
        corruption_file_path = data_dir + '/' + corruption + '.npy'
        if not os.path.isfile(corruption_file_path):
            download_gdrive(dict_gdrive_ids[corruption], corruption_file_path)
        images_all = np.load(corruption_file_path)
        images = images_all[(severity-1)*n_total_cifar : severity*n_total_cifar]
        n_img = int(np.ceil(n_examples/n_pert))
        x_test_list.append(images[:n_img])
        y_test_list.append(labels[:n_img])  # we need to duplicate the same labels potentially multiple times

    x_test, y_test = np.concatenate(x_test_list), np.concatenate(y_test_list)
    if shuffle:
        rand_idx = np.random.permutation(np.arange(len(x_test)))
        x_test, y_test = x_test[rand_idx], y_test[rand_idx]

    x_test = np.transpose(x_test, (0, 3, 1, 2))  # to make it in the pytorch format
    x_test = x_test.astype(np.float32) / 255  # to be compatible with our models
    x_test = torch.tensor(x_test)[:n_examples]  # to make sure that we get exactly n_examples but not a few samples more
    y_test = torch.tensor(y_test)[:n_examples]

    return x_test.cuda(), y_test.cuda()


if __name__ == '__main__':
    x_test, y_test = load_cifar10c(100, perturbations=['fog'])


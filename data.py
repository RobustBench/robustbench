import torch
import torchvision.datasets as datasets
import torch.utils.data as data
import torchvision.transforms as transforms


def load_cifar10(data_dir, n_ex, batch_size):
    transform_chain = transforms.Compose([transforms.ToTensor()])
    item = datasets.CIFAR10(root=data_dir, train=False, transform=transform_chain, download=True)
    test_loader = data.DataLoader(item, batch_size=batch_size, shuffle=False, num_workers=0)

    x_test = torch.cat([x for (x, y) in test_loader], 0)[:n_ex].cuda()
    y_test = torch.cat([y for (x, y) in test_loader], 0)[:n_ex].cuda()

    return x_test, y_test
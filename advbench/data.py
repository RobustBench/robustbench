import torch
import torchvision.datasets as datasets
import torch.utils.data as data
import torchvision.transforms as transforms


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


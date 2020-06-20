import argparse
import torch
import torchvision.datasets as datasets
import torch.utils.data as data
import torchvision.transforms as transforms

from utils import load_model


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='carmon_et_al_2019')
    parser.add_argument('--eps', type=float, default=8/255)
    parser.add_argument('--n_ex', type=int, default=100, help='number of examples to evaluate on')
    parser.add_argument('--batch_size', type=int, default=500, help='batch size for evaluation')
    parser.add_argument('--data_dir', type=str, default='./data', help='where to store downloaded datasets')
    parser.add_argument('--model_dir', type=str, default='./models', help='where to store downloaded models')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    model = load_model(args.model_name, args.model_dir).cuda().eval()

    # TODO: compress data loading
    transform_list = [transforms.ToTensor()]
    transform_chain = transforms.Compose(transform_list)
    item = datasets.CIFAR10(root=args.data_dir, train=False, transform=transform_chain, download=True)
    test_loader = data.DataLoader(item, batch_size=1000, shuffle=False, num_workers=0)
    # TODO: think how to implement input normalization -- directly in the carmon_et_al_2019 network?

    x_test = torch.cat([x for (x, y) in test_loader], 0)[:args.n_ex].cuda()
    y_test = torch.cat([y for (x, y) in test_loader], 0)[:args.n_ex].cuda()
    # TODO: maybe rework it to be a usual loader. or make the predict function batched

    acc = torch.mean((model(x_test).argmax(1) == y_test).float())
    print('accuracy={:.2%}'.format(acc))

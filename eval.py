import argparse
import torch
from utils import load_model
from data import load_cifar10
from model_zoo.model_utils import *


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='carmon2019unlabeled')
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
    x_test, y_test = load_cifar10(args.data_dir, args.n_ex, args.batch_size)

    # TODO: make the predict function in batches
    #acc = torch.mean((model(x_test).argmax(1) == y_test).float())
    acc = clean_accuracy(model, x_test, y_test, bs=128)
    print('clean accuracy: {:.2%}'.format(acc))
    # TODO: add AutoAttack


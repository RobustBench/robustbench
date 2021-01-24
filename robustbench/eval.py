import argparse

import torch
from autoattack import AutoAttack

from robustbench.model_zoo.enums import BenchmarkDataset, ThreatModel
from robustbench.utils import load_model, clean_accuracy
from robustbench.data import load_cifar10


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='Carmon2019Unlabeled')
    parser.add_argument('--threat_model', type=str, default='Linf', choices=[x.value for x in ThreatModel])
    parser.add_argument('--dataset', type=str, default='cifar10', choices=[x.value for x in BenchmarkDataset])
    parser.add_argument('--eps', type=float, default=8/255)
    parser.add_argument('--n_ex', type=int, default=100, help='number of examples to evaluate on')
    parser.add_argument('--batch_size', type=int, default=500, help='batch size for evaluation')
    parser.add_argument('--data_dir', type=str, default='./data', help='where to store downloaded datasets')
    parser.add_argument('--model_dir', type=str, default='./models', help='where to store downloaded models')
    parser.add_argument('--device', type=str, default='cuda:0', help='device to use for computations')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    device = torch.device(args.device)

    x_test, y_test = load_cifar10(args.n_ex, args.data_dir)
    x_test, y_test = x_test.to(device), y_test.to(device)
    model = load_model(args.model_name, args.model_dir, args.dataset, args.threat_model).to(device).eval()

    acc = clean_accuracy(model, x_test, y_test, batch_size=args.batch_size, device=device)
    print('Clean accuracy: {:.2%}'.format(acc))

    adversary = AutoAttack(model, norm=args.threat_model, eps=args.eps, version='standard', device=device)
    x_adv = adversary.run_standard_evaluation(x_test, y_test)

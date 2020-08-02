import argparse
from utils import load_model, clean_accuracy
from data import load_cifar10


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='Carmon2019Unlabeled')
    parser.add_argument('--eps', type=float, default=8/255)
    parser.add_argument('--n_ex', type=int, default=100, help='number of examples to evaluate on')
    parser.add_argument('--batch_size', type=int, default=500, help='batch size for evaluation')
    parser.add_argument('--data_dir', type=str, default='./data', help='where to store downloaded datasets')
    parser.add_argument('--model_dir', type=str, default='./models', help='where to store downloaded models')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()

    x_test, y_test = load_cifar10(args.n_ex, args.data_dir)
    model = load_model(args.model_name, args.model_dir).cuda().eval()

    acc = clean_accuracy(model, x_test, y_test, batch_size=args.batch_size)
    print('Clean accuracy: {:.2%}'.format(acc))
    # TODO: add AutoAttack, use args.eps


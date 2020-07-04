import sys
sys.path.append('.')
from tests.config import parse_args
from utils import load_model
from data import load_cifar10
from model_zoo.models import model_dicts
from model_zoo.model_utils import *


if __name__ == '__main__':
    args = parse_args()
    n_ex = 200

    models = model_dicts.keys()
    x_test, y_test = load_cifar10(args.data_dir, n_ex, args.batch_size)
    
    for model_name in models:
        model = load_model(model_name, args.model_dir).cuda().eval()

        acc = clean_accuracy(model, x_test, y_test, bs=args.batch_size)

        print('{} - clean accuracy {:.2%} (on {} examples)'.format(model_name, acc, n_ex))
        assert round(acc * 100., 2) > 70.0


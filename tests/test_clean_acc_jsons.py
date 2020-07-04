import sys
import json
sys.path.append('.')
from tests.config import parse_args
from utils import load_model
from data import load_cifar10
from model_zoo.models import model_dicts
from model_zoo.model_utils import *


if __name__ == '__main__':
    args = parse_args()
    n_ex = 10000

    models = model_dicts.keys()
    x_test, y_test = load_cifar10(args.data_dir, n_ex, args.batch_size)
    
    for model_name in models:
        model = load_model(model_name, args.model_dir).cuda().eval()

        acc = clean_accuracy(model, x_test, y_test, bs=args.batch_size)
        with open('./model_info/{}.json'.format(model_name), 'r') as model_info:
            data = json.load(model_info)

        print('{} - clean accuracy {:.2%}'.format(model_name, acc))
        assert round(acc * 100., 2) == float(data['clean_acc'])


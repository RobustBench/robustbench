import argparse
import torch
from utils import load_model
from data import load_cifar10
from model_zoo.model_utils import *
import json


models = ['Carmon2019Unlabeled',
    'Sehwag2020Hydra',
    'Wang2020Improving',
    'Hendrycks2019Using',
    'Rice2020Overfitting',
    'Zhang2019Theoretically',
    'Engstrom2019Robustness']

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
    x_test, y_test = load_cifar10(args.data_dir, args.n_ex, args.batch_size)
    
    for model_name in models:
        model = load_model(model_name, args.model_dir).cuda().eval()
        
    
        
        acc = clean_accuracy(model, x_test, y_test, bs=128)
        
        with open('./model_info/{}.json'.format(model_name), 'r') as model_info:
            data = json.load(model_info)

        assert acc * 100. == float(data['clean_acc'])
        print('{} - clean accuracy {:.2f}'.format(model_name, acc * 100.))
        

import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=500, help='batch size for evaluation')
    parser.add_argument('--data_dir', type=str, default='../data', help='where to store downloaded datasets')
    parser.add_argument('--model_dir', type=str, default='../models', help='where to store downloaded models')
    args = parser.parse_args()
    return args


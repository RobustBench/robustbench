import unittest
import sys
from tests.config import parse_args
from advbench.utils import load_model, clean_accuracy
from advbench.data import load_cifar10
from advbench.model_zoo.models import model_dicts



class CleanAccTester(unittest.TestCase):
    def test_clean_acc_jsons_exact():
        args = parse_args()
        n_ex = 200

        models = model_dicts.keys()
        x_test, y_test = load_cifar10(n_ex, args.data_dir)

        n_tests_passed = 0
        for model_name in models:
            model = load_model(model_name, args.model_dir).cuda().eval()

            acc = clean_accuracy(model, x_test, y_test, batch_size=args.batch_size)

            success = round(acc * 100., 2) > 70.0
            n_tests_passed += success
            print('{}: clean accuracy {:.2%} (on {} examples), test passed: {}'.format(model_name, acc, n_ex, success))

        print('Test is passed for {}/{} models.'.format(n_tests_passed, len(models)))


    def test_clean_acc_jsons_exact():
        args = parse_args()
        n_ex = 10000

        models = model_dicts.keys()
        x_test, y_test = load_cifar10(n_ex, args.data_dir)

        n_tests_passed = 0
        for model_name in models:
            model = load_model(model_name, args.model_dir).cuda().eval()

            acc = clean_accuracy(model, x_test, y_test, batch_size=args.batch_size)
            with open('./model_info/{}.json'.format(model_name), 'r') as model_info:
                json_dict = json.load(model_info)

            success = round(acc * 100., 2) == float(json_dict['clean_acc'])
            n_tests_passed += success
            print('{}: clean accuracy {:.2%}, test passed: {}'.format(model_name, acc, success))

        print('Test is passed for {}/{} models.'.format(n_tests_passed, len(models)))



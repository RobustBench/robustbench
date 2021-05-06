import json
import unittest
from pathlib import Path
from typing import Callable

import torch

from robustbench.data import load_clean_dataset
from robustbench.model_zoo.models import model_dicts
from robustbench.utils import clean_accuracy, load_model, BenchmarkDataset
from tests.config import get_test_config
from tests.utils_testing import slow


def _accuracy_computation(success_criterion: Callable[[str, float, str, str], bool], n_ex: int) -> None:
    config = get_test_config()
    device = torch.device(config["device"])

    tot_models = 0
    n_tests_passed = 0

    for dataset, dataset_dict in model_dicts.items():
        if dataset == BenchmarkDataset("cifar10"): continue
        print(f"Test models trained on {dataset.value}")
        x_test, y_test = load_clean_dataset(dataset, n_ex, config["data_dir"])

        for threat_model, threat_model_dict in dataset_dict.items():
            print(f"Test models robust wrt {threat_model.value}")
            models = list(threat_model_dict.keys())
            tot_models += len(models)

            for model_name in models:
                model = load_model(model_name, config["model_dir"],
                                   dataset, threat_model).to(device)
                model.eval()
                acc = clean_accuracy(model, x_test, y_test,
                                     batch_size=config["batch_size"], device=device)

                success = success_criterion(model_name, acc, dataset.value, threat_model.value)
                n_tests_passed += int(success)
                print(f"{model_name}: clean accuracy {acc:.2%} (on {n_ex} examples),"
                      f" test passed: {success}")

    print(f"Test is passed for {n_tests_passed}/{tot_models} models.")


class CleanAccTester(unittest.TestCase):

    def test_clean_acc_jsons_fast(self):
        datasets_acc = {
            "cifar10": 70.0,
            "cifar100": 45.0
        }
        def fast_acc_success_criterion(model_name: str, acc: float, dataset: str, threat_model: str) -> bool:
            self.assertGreater(round(acc * 100., 2), datasets_acc[dataset])
            return round(acc * 100., 2) > datasets_acc[dataset]

        n_ex = 200
        _accuracy_computation(fast_acc_success_criterion, n_ex)

    @slow
    def test_clean_acc_jsons_exact(self):
        def exact_acc_success_criterion(model_name: str, acc: float, dataset: str, threat_model: str) -> bool:
            model_info_path = Path("model_info") / dataset / threat_model / f"{model_name}.json"

            with open(model_info_path) as model_info:
                json_dict = json.load(model_info)

            self.assertLessEqual(abs(round(acc * 100., 2) - float(json_dict['clean_acc'])), 0.05)

            return abs(round(acc * 100., 2) - float(json_dict['clean_acc'])) <= 0.05

        n_ex = 10000
        _accuracy_computation(exact_acc_success_criterion, n_ex)

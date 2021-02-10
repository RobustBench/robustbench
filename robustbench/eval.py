import argparse
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple, Union

import pandas as pd
import torch
from tqdm import tqdm
from autoattack import AutoAttack
from torch import nn

from robustbench.data import CORRUPTIONS, load_clean_dataset, load_corruptions_dataset
from robustbench.model_zoo.enums import BenchmarkDataset, ThreatModel
from robustbench.utils import clean_accuracy, update_json


def benchmark(model: Union[nn.Module, Sequence[nn.Module]],
              n_examples: int,
              dataset: Union[str,
                             BenchmarkDataset] = BenchmarkDataset.cifar_10,
              threat_model: Union[str, ThreatModel] = ThreatModel.Linf,
              to_disk: bool = False,
              model_name: Optional[str] = None,
              data_dir: str = "./data",
              device: Optional[Union[torch.device,
                                     Sequence[torch.device]]] = None,
              batch_size: int = 32,
              eps: Optional[float] = None) -> Tuple[float, float]:
    """Benchmarks the given model(s).

    It is possible to benchmark on 3 different threat models, and to save the results on disk. In
    the future benchmarking multiple models in parallel is going to be possible.

    :param model: The model to benchmark.
    :param n_examples: The number of examples to use to benchmark the model.
    :param dataset: The dataset to use to benchmark. Must be one of {cifar10, cifar100}
    :param threat_model: The threat model to use to benchmark, must be one of {L2, Linf
    corruptions}
    :param to_disk: Whether the results must be saved on disk as .json.
    :param model_name: The name of the model to use to save the results. Must be specified if
    to_json is True.
    :param data_dir: The directory where the dataset is or where the dataset must be downloaded.
    :param device: The device to run the computations.
    :param batch_size: The batch size to run the computations. The larger, the faster the
    evaluation.
    :param eps: The epsilon to use for L2 and Linf threat models. Must not be specified for
    corruptions threat model.

    :return: A Tuple with the clean accuracy and the accuracy in the given threat model.
    """
    if isinstance(model, Sequence) or isinstance(device, Sequence):
        # Multiple models evaluation in parallel not yet implemented
        raise NotImplementedError

    dataset_: BenchmarkDataset = BenchmarkDataset(dataset)
    threat_model_: ThreatModel = ThreatModel(threat_model)

    device = device or torch.device("cpu")

    clean_x_test, clean_y_test = load_clean_dataset(dataset_, n_examples,
                                                    data_dir)
    clean_x_test, clean_y_test = clean_x_test.to(device), clean_y_test.to(
        device)

    accuracy = clean_accuracy(model,
                              clean_x_test,
                              clean_y_test,
                              batch_size=batch_size,
                              device=device)
    print(f'Clean accuracy: {accuracy:.2%}')

    if threat_model_ in {ThreatModel.Linf, ThreatModel.L2}:
        if eps is None:
            raise ValueError(
                "If the threat model is L2 or Linf, `eps` must be specified.")

        adversary = AutoAttack(model,
                               norm=threat_model_.value,
                               eps=eps,
                               version='standard',
                               device=device)
        x_adv = adversary.run_standard_evaluation(clean_x_test, clean_y_test)
        adv_accuracy = clean_accuracy(model,
                                      x_adv,
                                      clean_y_test,
                                      batch_size=batch_size,
                                      device=device)
        print(f'Adversarial accuracy: {adv_accuracy:.2%}')
    elif threat_model_ == ThreatModel.corruptions:
        print(f"Evaluating over {len(CORRUPTIONS)} corruptions")
        # Save into a dict to make a Pandas DF with nested index
        adv_accuracy = corruptions_evaluation(batch_size, data_dir, dataset_,
                                              device, model, n_examples,
                                              to_disk, model_name)
    else:
        raise NotImplementedError

    update_json(dataset_, threat_model_, model_name, accuracy, adv_accuracy, eps)

    return accuracy, adv_accuracy


def corruptions_evaluation(batch_size: int, data_dir: str,
                           dataset: BenchmarkDataset, device: torch.device,
                           model: nn.Module, n_examples: int, to_disk: bool,
                           model_name: Optional[str]) -> float:

    if to_disk and model_name is None:
        raise ValueError(
            "If `to_disk` is True, `model_name` should be specified.")

    model_results_dict: Dict[Tuple[str, int], float] = {}
    for corruption in tqdm(CORRUPTIONS):
        for severity in range(1, 6):
            x_corrupt, y_corrupt = load_corruptions_dataset(
                dataset,
                n_examples,
                severity,
                data_dir,
                corruptions=[corruption])

            corruption_severity_accuracy = clean_accuracy(
                model,
                x_corrupt,
                y_corrupt,
                batch_size=batch_size,
                device=device)

            model_results_dict[(corruption,
                                severity)] = corruption_severity_accuracy

    model_results = pd.DataFrame(model_results_dict, index=[model_name])
    adv_accuracy = model_results.values.mean()

    if not to_disk:
        return adv_accuracy

    # Save unaggregated results on disk
    existing_results_path = Path(
        "model_info") / dataset.value / "corruptions" / "unagg_results.csv"
    try:
        existing_results = pd.read_csv(existing_results_path,
                                       header=[0, 1],
                                       index_col=0)
        existing_results.columns = existing_results.columns.set_levels([
            existing_results.columns.levels[0],
            existing_results.columns.levels[1].astype(int)
        ])
        full_results = pd.concat([existing_results, model_results])
    except FileNotFoundError:
        full_results = model_results
    full_results.to_csv(existing_results_path)

    return adv_accuracy


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name',
                        type=str,
                        default='Carmon2019Unlabeled')
    parser.add_argument('--threat_model',
                        type=str,
                        default='Linf',
                        choices=[x.value for x in ThreatModel])
    parser.add_argument('--dataset',
                        type=str,
                        default='cifar10',
                        choices=[x.value for x in BenchmarkDataset])
    parser.add_argument('--eps', type=float, default=8 / 255)
    parser.add_argument('--n_ex',
                        type=int,
                        default=100,
                        help='number of examples to evaluate on')
    parser.add_argument('--batch_size',
                        type=int,
                        default=500,
                        help='batch size for evaluation')
    parser.add_argument('--data_dir',
                        type=str,
                        default='./data',
                        help='where to store downloaded datasets')
    parser.add_argument('--model_dir',
                        type=str,
                        default='./models',
                        help='where to store downloaded models')
    parser.add_argument('--device',
                        type=str,
                        default='cuda:0',
                        help='device to use for computations')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    ...

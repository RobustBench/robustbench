from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from numpy.typing import ArrayLike
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from robustbench.data import load_clean_dataset
from robustbench.eval.utils import check_model_eval
from robustbench.model_zoo.architectures.utils import LipschitzModel
from robustbench.model_zoo.enums import BenchmarkDataset

RESULTS_FILENAME = "lipschitz_results.txt"


def box(x_prime: torch.Tensor, x: torch.Tensor, eps: float) -> torch.Tensor:
    min_x = torch.max(torch.zeros_like(x), x - eps)
    max_x = torch.min(torch.ones_like(x), x + eps)

    return torch.max(min_x, torch.min(x_prime, max_x))


def lipschitz_loss(model: nn.Module, normalization: Optional[str], p,
                   x_1: torch.Tensor, x_2: torch.Tensor) -> torch.Tensor:
    out_1 = torch.flatten(model(x_1), start_dim=1)
    out_2 = torch.flatten(model(x_2), start_dim=1)
    if normalization == "l2":
        out_1 = F.normalize(out_1)
        out_2 = F.normalize(out_2)
    if normalization == "mean_logit":
        out_1 /= out_1.mean()
        out_2 /= out_2.mean()
    numerator = (out_1 - out_2).norm(dim=1, p=1)
    flattened_x_1 = torch.flatten(x_1, start_dim=1)
    flattened_x_2 = torch.flatten(x_2, start_dim=1)
    # Add 1e-9 for numerical stability
    denominator = (flattened_x_1 - flattened_x_2).norm(p=p, dim=1) + 1e-9
    return (numerator / denominator).mean()


def compute_lipschitz_batch(
    model: nn.Module,
    x: torch.Tensor,
    eps: float,
    step_size: float,
    n_steps: int,
    normalization: Optional[str],
    p: float,
    summary_writer_suffix: Optional[Tuple[SummaryWriter, str]] = None
) -> Tuple[float, Tuple[ArrayLike, ArrayLike]]:
    """Computes local (i.e. eps-ball) Lipschitzness of the given `model` on a batch of data."""
    valid_normalizations = {None, "l2", "mean_logit"}
    if normalization not in valid_normalizations:
        raise ValueError(
            f"`normalization` must be one of {valid_normalizations}")

    # Initialize to a slightly different random value
    x_1 = box(x.clone() + step_size * torch.randn_like(x), x,
              eps).requires_grad_(True)
    x_2 = x.clone().requires_grad_(True)

    # Only L2 normalization should be kept in consideration for optimization
    optim_norm = "l2" if normalization == "l2" else None
    max_lips = lipschitz_loss(model, optim_norm, p, x_1, x_2).item()
    max_x_1, max_x_2 = x_1.detach(), x_2.detach()

    for i in range(n_steps):
        curr_lips = lipschitz_loss(model, optim_norm, p, x_1, x_2)
        if curr_lips.item() > max_lips:
            max_lips = curr_lips.item()
            max_x_1, max_x_2 = x_1.detach(), x_2.detach()

        curr_lips.backward()
        x_1 = box(x_1.detach() + step_size * x_1.grad.sign(), x,
                  eps).requires_grad_(True)
        x_2 = box(x_2.detach() + step_size * x_2.grad.sign(), x,
                  eps).requires_grad_(True)

        if summary_writer_suffix is not None:
            summary_writer, suffix = summary_writer_suffix
            summary_writer.add_scalar(f"lipschitz_{suffix}", curr_lips.item(),
                                      i)

    final_lips = lipschitz_loss(model, optim_norm, p, x_1, x_2).item()
    if final_lips > max_lips:
        max_lips = final_lips
        max_x_1, max_x_2 = x_1.detach(), x_2.detach()

    if normalization == "mean_logit":
        max_lips = lipschitz_loss(model, normalization, p, max_x_1,
                                  max_x_2).item()

    return max_lips, (max_x_1.cpu().numpy(), max_x_2.cpu().numpy())


def compute_lipschitz(
    model: nn.Module,
    dl: DataLoader,
    eps: float,
    step_size: float,
    n_steps: int = 50,
    normalization: Optional[str] = None,
    p: float = float("inf"),
    device: Optional[torch.device] = None,
    summary_writer_suffix: Optional[Tuple[SummaryWriter, str]] = None
) -> Tuple[float, Tuple[ArrayLike, ArrayLike]]:
    """Computes local (i.e. eps-ball) Lipschitzness of the given `model` around each sample.

    The computation is done via a PDG-like procedure, where we optimized both the inputs of the
    Lipschitz loss.

    :param model: The layer whose Lipschitzness has to be computed.
    :param dl: The dataloader of data to compute Lipschitzness on.
    :param eps: The ball boundary (around each sample)
    :param step_size: The step size of the each step.
    :param n_steps: The number of steps to run.
    :param normalization: The normalization to apply to the each layer's output (default None).
    :param p: the p of the norm of Lipschitzness.
    :param device: The device to run computations.
    :param summary_writer_suffix:

    :return: The local Lipschitz constant, and the inputs found for the model.
    """
    device = device or torch.device("cpu")
    model_dev = model.to(device)

    lips = 0.
    x_1s, x_2s = [], []
    prog_bar = tqdm(dl)

    for i, (x, _) in enumerate(prog_bar):
        x_dev = x.to(device)
        batch_lips, (x_1, x_2) = compute_lipschitz_batch(
            model_dev, x_dev, eps, step_size, n_steps, normalization, p,
            summary_writer_suffix)
        lips += batch_lips
        x_1s.append(x_1)
        x_2s.append(x_2)
        prog_bar.set_postfix({"lips": lips / (i + 1)})

    return lips / len(dl), (np.stack(x_1s), np.stack(x_2s))


def benchmark_lipschitz(
    model: LipschitzModel,
    n_examples: int = 10_000,
    dataset: Union[str, BenchmarkDataset] = BenchmarkDataset.cifar_10,
    data_dir: str = "./data",
    batch_size: int = 16,
    eps: float = 8 / 255,
    step_size: float = (8 / 255) / 5,
    n_steps: int = 50,
    normalization: Optional[str] = None,
    p: float = float("inf"),
    device: Optional[torch.device] = None,
    logging_dir: Optional[Path] = None,
    model_name: Optional[str] = None,
) -> Tuple[List[float], List[Tuple[ArrayLike, ArrayLike]]]:
    """Benchmarks the Lipschitzness of a given model and dataset, at the layers given
    by the `get_lipschitz_layers()` method.

    The computation is done via a PDG-like procedure, where we optimized both the inputs of the
    Lipschitz loss.

    The `get_lipschitz_layers` must return a sequence of layers at which the model must be
    evaluated. This function iterates over layers by stacking them after each iterations. E.g.
    a model with 2 subsequent linear layers `l1` and `l2` should return `[l1, l2]`. Check out
    `robustbench.model_zoo.architectures.wide_resnet.WideResNet` to see an example.

    :param model_name:
    :param model: The model to benchmark. It must implement the `get_lipschitz_layers()` method.
    :param n_examples: The number of examples to use for the benchmark.
    :param dataset: The dataset of the benchmark.
    :param data_dir: The directory where the data should be downloaded.
    :param batch_size: The batch size to use.
    :param eps: The radius of the ball where to look.
    :param step_size: The step-size of the PGD-based procedure.
    :param n_steps: The number of PGD steps.
    :param normalization: What kind of normalization to apply. None by default.
    :param p: The type of norm of the denominator.
    :param device: The device to run computations.
    :param logging_dir: The directory to save the logs.
    :return: The computed local Lipschitz constant, and the inputs found for each layer..
    """
    dataset_ = BenchmarkDataset(dataset)
    check_model_eval(model)

    x, y = load_clean_dataset(dataset_, n_examples, data_dir)
    ds = TensorDataset(x, y)
    dl = DataLoader(ds, batch_size=batch_size, num_workers=8, shuffle=False)

    lips = []
    inputs = []

    if logging_dir is not None:
        assert model_name is not None
        tensorboard_dir = (
            logging_dir / model_name /
            f"{normalization}_{eps:.2f}_{n_steps}_{step_size:.2f}")
        if not tensorboard_dir.exists():
            tensorboard_dir.mkdir(parents=True)
        summary_writer = SummaryWriter(str(tensorboard_dir))
        results_filename = logging_dir / RESULTS_FILENAME
    else:
        summary_writer = None
        results_filename = None
        tensorboard_dir = None

    net = nn.Sequential()
    for i, layer in enumerate(model.get_lipschitz_layers()):
        net = nn.Sequential(*net, layer)
        suffix = f"layer_{i}"
        if tensorboard_dir is not None:
            tensorboard_dir_suffix = summary_writer, suffix
        else:
            tensorboard_dir_suffix = None
        layer_lips, layer_inputs = compute_lipschitz(net, dl, eps, step_size,
                                                     n_steps, normalization, p,
                                                     device,
                                                     tensorboard_dir_suffix)
        if logging_dir is not None:
            string_to_log = f"{model_name},{dataset_.value},{i},{p},{eps},{step_size},{n_steps},{normalization},{layer_lips}\n"
            with open(results_filename, "a") as fp:
                fp.write(string_to_log)

        lips.append(layer_lips)
        inputs.append(layer_inputs)

    if tensorboard_dir is not None and summary_writer is not None:
        hparams = {
            "model": model_name,
            "eps": eps,
            "step_size": step_size,
            "normalization": str(normalization),
            "n_steps": n_steps,
            "p": p
        }
        metrics = {f"lipschitz_{i}": l for i, l in enumerate(lips)}
        print(hparams)
        print(metrics)
        summary_writer.add_hparams(hparams, metrics)
        np.savez(tensorboard_dir / f"inputs", inputs)

    return lips, inputs

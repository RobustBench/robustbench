from typing import Optional, Union

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from robustbench.data import load_clean_dataset
from robustbench.eval.utils import check_model_eval
from robustbench.model_zoo.architectures.utils import LipschitzModel
from robustbench.model_zoo.enums import BenchmarkDataset


def box(x_prime: torch.Tensor, x: torch.Tensor, eps: float) -> torch.Tensor:
    min_x = torch.max(torch.zeros_like(x), x - eps)
    max_x = torch.min(torch.ones_like(x), x + eps)

    return torch.max(min_x, torch.min(x_prime, max_x))


def compute_lipschitz_batch(model: nn.Module, x: torch.Tensor, eps: float,
                            step_size: float, n_steps: int, normalization: Optional[str],
                            p: float) -> float:
    """Computes local (i.e. eps-ball) Lipschitzness of the given `model` on a batch of data."""
    valid_normalizations = {None, "l2", "avg_logit"}
    if normalization not in valid_normalizations:
        raise ValueError(f"`normalization` must be one of {valid_normalizations}")

    # Function to optimize
    def lipschitz(x_1_: torch.Tensor, x_2_: torch.Tensor) -> torch.Tensor:
        out_1 = torch.flatten(model(x_1_), start_dim=1)
        out_2 = torch.flatten(model(x_2_), start_dim=1)
        if normalization == "l2":
            out_1 = F.normalize(out_1)
            out_2 = F.normalize(out_2)
        if normalization == "avg_logit":
            out_1 /= out_1.mean(dim=1)
            out_2 /= out_2.mean(dim=1)
        numerator = (out_1 - out_2).norm(dim=1, p=1)
        flattened_x_1_ = torch.flatten(x_1_, start_dim=1)
        flattened_x_2_ = torch.flatten(x_2_, start_dim=1)
        # Add 1e-9 for numerical stability
        denominator = (flattened_x_1_ - flattened_x_2_).norm(p=p, dim=1) + 1e-9
        return (numerator / denominator).mean()

    # Initialize to a slightly different random value
    x_1 = box(x.clone() + step_size * torch.randn_like(x), x,
              eps).requires_grad_(True)
    x_2 = x.clone().requires_grad_(True)

    max_lips = lipschitz(x_1, x_2).item()

    for i in range(n_steps):
        y = lipschitz(x_1, x_2)
        y.backward()
        x_1 = box(x_1.detach() + step_size * x_1.grad.sign(), x,
                  eps).requires_grad_(True)
        x_2 = box(x_2.detach() + step_size * x_2.grad.sign(), x,
                  eps).requires_grad_(True)
        max_lips = max(max_lips, y.item())

    return max(max_lips, lipschitz(x_1, x_2).item())


def compute_lipschitz(
        model: nn.Module,
        dl: DataLoader,
        eps: float,
        step_size: float,
        n_steps: int = 50,
        normalization: Optional[str] = None,
        p: float = float("inf"),
        device: Optional[torch.device] = None,
):
    """Computes local (i.e. eps-ball) Lipschitzness of the given `model` around each sample.

    :param model: The layer whose Lipschitzness has to be computed.
    :param dl: The dataloader of data to compute Lipschitzness on.
    :param eps: The ball boundary (around each sample)
    :param step_size: The step size of the each step.
    :param n_steps: The number of steps to run.
    :param normalization: The normalization to apply to the each layer's output (default None).
    :param p: the p of the norm of Lipschitzness.
    :param device: The device to run computations.

    :return: The local Lipschitz constant.
    """
    device = device or torch.device("cpu")
    model_dev = model.to(device)

    lips = 0.
    prog_bar = tqdm(dl)

    for i, (x, _) in enumerate(prog_bar):
        x_dev = x.to(device)
        batch_lips = compute_lipschitz_batch(model_dev, x_dev, eps, step_size, n_steps,
                                             normalization, p)
        lips += batch_lips
        prog_bar.set_postfix({"lips": lips / (i + 1)})

    return lips / len(dl)


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
        device: Optional[torch.device] = None):
    dataset_ = BenchmarkDataset(dataset)
    check_model_eval(model)

    x, y = load_clean_dataset(dataset_, n_examples, data_dir)
    dataset = TensorDataset(x, y)
    dl = DataLoader(dataset,
                    batch_size=batch_size,
                    num_workers=8,
                    shuffle=True)

    lips = []

    net = nn.Sequential()

    for layer in model.get_lipschitz_layers():
        net = nn.Sequential(*net, layer)
        layer_lips = compute_lipschitz(net, dl, eps, step_size, n_steps,
                                       normalization, p, device)
        lips.append(layer_lips)

    return lips

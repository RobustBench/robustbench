from typing import Optional, Union

import torch
import torch.nn.functional as F
from torch import nn
from torch.autograd import grad
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
                            step_size: float, n_steps: int, l2_normalize: bool,
                            p: float) -> float:
    """Computes local (i.e. eps-ball) Lipschitzness of the given `model` on a batch of data."""
    def grad_norm(x_prime_: torch.Tensor) -> torch.Tensor:
        x_prime_ = torch.flatten(x_prime_, start_dim=1)
        if l2_normalize:
            x_prime_ = F.normalize(x_prime_)
        model_output = model(x_prime_).sum()
        model_grad = grad(model_output, x_prime_, create_graph=True)[0]

        return torch.norm(model_grad, p=p)

    # Initialize to a slightly different random value
    x_prime = box(x + step_size * torch.randn_like(x), x,
                  eps).requires_grad_(True)
    max_lips = grad_norm(x_prime).mean().item()

    for i in range(n_steps):
        y = grad_norm(x_prime).mean()
        # The gradient can be independent of the input if the model is linear
        # if this is the case, then make it 0.
        gradient = grad(y, x_prime, allow_unused=True)[0]
        if gradient is None:
            gradient = torch.zeros_like(x_prime)

        x_prime = box(x_prime.detach() + step_size * gradient.sign(), x,
                      eps).requires_grad_(True)
        max_lips = max(max_lips, y.item())

    return max(max_lips, grad_norm(x_prime).mean().item())


def compute_lipschitz(
        model: nn.Module,
        dl: DataLoader,
        eps: float,
        step_size: float,
        n_steps: int = 50,
        l2_normalize: bool = True,
        p: float = float("inf"),
        device: Optional[torch.device] = None,
):
    """Computes local (i.e. eps-ball) Lipschitzness of the given `model` around each sample.

    :param model: The layer whose Lipschitzness has to be computed.
    :param dl: The dataloader of data to compute Lipschitzness on.
    :param eps: The ball boundary (around each sample)
    :param step_size: The step size of the each step.
    :param n_steps: The number of steps to run.
    :param l2_normalize: Whether the logits should be projected to the unit L2 ball.
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
        batch_lips = compute_lipschitz_batch(model_dev, x_dev, eps, step_size,
                                             n_steps, l2_normalize, p)
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
        step_size: float = (8 / 255) / 3,
        n_steps: int = 50,
        l2_normalize: bool = True,
        p: float = float("inf"),
        device: Optional[torch.device] = None):
    dataset_ = BenchmarkDataset(dataset)
    check_model_eval(model)

    x, y = load_clean_dataset(dataset_, n_examples, data_dir)
    dataset = TensorDataset(x, y)
    dl = DataLoader(dataset,
                    batch_size=batch_size,
                    drop_last=True,
                    num_workers=8)

    lips = []

    net = nn.Sequential()

    for layer in model.get_lipschitz_layers():
        net = nn.Sequential(*net, layer)
        layer_lips = compute_lipschitz(net, dl, eps, step_size, n_steps,
                                       l2_normalize, p, device)
        lips.append(layer_lips)

    return lips

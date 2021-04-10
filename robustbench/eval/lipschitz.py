from typing import Optional, Union

import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from robustbench.data import load_clean_dataset
from robustbench.eval.utils import check_model_eval
from robustbench.model_zoo.architectures.utils import Layer, LipschitzModel
from robustbench.model_zoo.enums import BenchmarkDataset


def box(x_prime: torch.Tensor, x: torch.Tensor, eps: float) -> torch.Tensor:
    min_x = torch.max(torch.zeros_like(x), x - eps)
    max_x = torch.min(torch.ones_like(x), x + eps)

    return torch.max(min_x, torch.min(x_prime, max_x))


def compute_lipschitz_batch(model: Layer,
                            x: torch.Tensor,
                            eps: float,
                            step_size: float,
                            n_steps: int = 50,
                            normalize_by_logits: bool = True) -> float:
    """Computes local (i.e. eps-ball) Lipschitzness of the given `model` on
    a batch of data.
    """

    # Function to differentiate
    def f(x_prime_: torch.Tensor) -> torch.Tensor:
        numerator = torch.norm(torch.flatten(model(x), start_dim=1) -
                               torch.flatten(model(x_prime_), start_dim=1),
                               p=1,
                               dim=1)
        denominator = torch.norm(torch.flatten(x, start_dim=1) -
                                 torch.flatten(x_prime_, start_dim=1),
                                 p=float("inf"),
                                 dim=1)
        return numerator / denominator

    # Initialize to a slightly different random value
    x_prime = box(x + step_size * torch.randn_like(x), x, eps)
    x_prime.requires_grad = True
    max_lips = f(x_prime).mean().item()

    for i in range(n_steps):
        y = f(x_prime).mean()
        max_lips = max(max_lips, y.item())
        y.backward()
        grads_x_prime = x_prime.grad
        x_prime = box(x_prime.detach() + step_size * grads_x_prime, x,
                      eps).requires_grad_(True)

    max_lips = max(max_lips, f(x_prime).mean().item())
    if normalize_by_logits:
        return max_lips / model(x).abs().mean().item()
    return max_lips


def compute_lipschitz(
    model: Layer,
    dl: DataLoader,
    eps: float,
    step_size: float,
    n_steps: int = 50,
    normalize_by_logits: bool = True,
    device: Optional[torch.device] = None,
):
    """Computes local (i.e. eps-ball) Lipschitzness of the given `model`.

    We use the method proposed by Yang et al. [1]_.

    .. [1] Yao-Yuan Yang, Cyrus Rashtchian, Hongyang Zhang, Russ R. Salakhutdinov,
        Kamalika Chaudhuri, A Closer Look at Accuracy vs. Robustness, NeurIPS 2020.

    :param model: The layer whose Lipschitzness has to be computed.
    :param dl: The dataloader of data to compute Lipschitzness on.
    :param eps: The ball boundary (around each sample)
    :param step_size: The step size of the each step.
    :param n_steps: The number of steps to run.
    :param normalize_by_logits: Whether the Lipschitz constant should be normalized by the
        output logits.
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
                                             n_steps, normalize_by_logits)
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
        normalize_by_logits: bool = True,
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

    for layer in model.get_lipschitz_layers():
        layer_lips = compute_lipschitz(layer, dl, eps, step_size, n_steps,
                                       normalize_by_logits, device)
        lips.append(layer_lips)

    return lips

import torch
from torch.autograd import functional as agf
from torch import nn


def box(x_prime: torch.Tensor, x: torch.Tensor, eps: float) -> torch.Tensor:
    min_x = torch.max(torch.zeros(x.shape), x - eps)
    max_x = torch.min(torch.ones(x.shape), x + eps)

    return torch.max(min_x, torch.min(x_prime, max_x))


def compute_lipschitz(model: nn.Module, x: torch.Tensor, eps: float, step_size: float,
                      n_steps: int = 50) -> float:
    def f(x_prime_: torch.Tensor) -> torch.Tensor:
        numerator = torch.norm(model(x) - model(x_prime_), p=1, dim=1)
        denominator = torch.norm(x - x_prime_, p=float("inf"), dim=1)
        # print((numerator / denominator).mean())
        return numerator / denominator

    x_prime = box(x + step_size * torch.randn_like(x), x, eps)

    for i in range(n_steps):
        jac_x_prime = agf.jacobian(f, x_prime, strict=True)
        grads_x_prime = torch.diagonal(jac_x_prime).t()
        # print(grads_x_prime)
        x_prime = box(x_prime + step_size * grads_x_prime, x, eps)

    return torch.mean(f(x_prime)).item()

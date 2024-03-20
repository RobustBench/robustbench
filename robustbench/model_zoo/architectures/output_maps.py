import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np


def normalize_topk(logits, k=None, center_only=False):
    logits = torch.clamp(logits, min=-1e15, max=1e15)
    if k is None or k >= logits.shape[1]:
        logits_topk = logits
    else:
        logits_topk = logits.topk(k=k, dim=1).values

    logits_mean = logits_topk.mean(dim=1).unsqueeze(dim=1)
    if center_only:
        return logits - logits_mean

    logits_var = logits_topk.var(dim=1).unsqueeze(dim=1)
    return (logits - logits_mean) / (logits_var + 1e-8).sqrt()


class IdentityMap(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, logits, return_probs=False):
        return logits.softmax(dim=1) if return_probs else logits


class HardMaxMap(nn.Module):
    def __init__(self, ln_k=250):
        super().__init__()
        self.approx = LNPowerScaleMap(scale=1e6, power=1, ln_k=ln_k, center_only=False)

    def forward_exact(self, logits, return_probs=False):
        probs = F.one_hot(logits.argmax(dim=1), num_classes=logits.shape[1]).float()
        mapped_output = probs if return_probs else (probs - 1e-12) * np.inf
        return mapped_output.cpu() if 'mps' in str(logits.device) else mapped_output

    def forward(self, logits, return_probs=False):
        if logits.grad_fn is not None:
            return self.approx(logits, return_probs)
        else:
            return self.forward_exact(logits, return_probs)


class ScaleMap(nn.Module):
    def __init__(self, scale=1):
        super().__init__()
        self.scale = nn.parameter.Parameter(torch.tensor(scale), requires_grad=False)

    def forward(self, logits, return_probs=False):
        scaled_logits = logits * self.scale
        mapped_output = scaled_logits.softmax(dim=1) if return_probs else scaled_logits
        return mapped_output.cpu() if 'mps' in str(logits.device) else mapped_output


class LayerNormMap(nn.Module):
    def __init__(self, ln_k=250, center_only=False):
        super().__init__()
        self.ln_k, self.center_only = ln_k, center_only

    def forward(self, logits, return_probs=False):
        orig_device = logits.device
        if 'cuda' not in str(orig_device):
            logits = logits.cpu()

        normed_logits = normalize_topk(logits.double(), k=self.ln_k, center_only=self.center_only)
        mapped_output = normed_logits.softmax(dim=1) if return_probs else normed_logits
        if 'mps' not in str(orig_device):
            return mapped_output.to(orig_device)
        else:
            return mapped_output


class LNPowerScaleMap(nn.Module):
    def __init__(self, scale=1, power=1, ln_k=250, center_only=False):
        super().__init__()
        self.ln_k, self.center_only = ln_k, center_only
        self.scale = nn.parameter.Parameter(torch.tensor(scale), requires_grad=False)
        self.power = nn.parameter.Parameter(torch.tensor(power), requires_grad=False)

    def forward(self, logits, return_probs=False):
        orig_device = logits.device
        if 'cuda' not in str(orig_device):
            logits = logits.cpu()
            power, scale = self.power.cpu(), self.scale.cpu()
        else:
            power, scale = self.power, self.scale

        # Normalize
        normed_logits = normalize_topk(logits.double(), k=self.ln_k, center_only=self.center_only)
        # Apply power
        powered_logits = normed_logits.abs() ** power * normed_logits.sign()
        # Apply scale
        scaled_logits = powered_logits * scale

        mapped_output = scaled_logits.softmax(dim=1) if return_probs else scaled_logits
        if 'mps' not in str(orig_device):
            return mapped_output.to(orig_device)
        else:
            return mapped_output


class LNClampPowerScaleMap(nn.Module):
    def __init__(
        self, scale=.4, power=1, clamp_bias=-1, clamp_fn=nn.GELU(), ln_k=250, center_only=False
    ):
        super().__init__()
        self.ln_k, self.center_only = ln_k, center_only
        self.scale = nn.parameter.Parameter(torch.tensor(scale), requires_grad=False)
        self.power = nn.parameter.Parameter(torch.tensor(power), requires_grad=False)
        self.clamp_bias = nn.parameter.Parameter(torch.tensor(clamp_bias), requires_grad=False)
        self.clamp_fn = clamp_fn

    def forward(self, logits, return_probs=False):
        # Normalize
        orig_device = logits.device
        if 'cuda' not in str(orig_device):
            logits = logits.cpu()
            power, scale, clamp_bias = self.power.cpu(), self.scale.cpu(), self.clamp_bias.cpu()
        else:
            power, scale, clamp_bias = self.power, self.scale, self.clamp_bias

        normed_logits = normalize_topk(logits.double(), k=self.ln_k, center_only=self.center_only)
        # Apply clamping function
        clamped_logits = self.clamp_fn(normed_logits + clamp_bias)
        # Apply power
        powered_logits = clamped_logits.abs() ** power * clamped_logits.sign()
        # Apply scale
        scaled_logits = powered_logits * scale

        mapped_output = scaled_logits.softmax(dim=1) if return_probs else scaled_logits
        if 'mps' not in str(orig_device):
            return mapped_output.to(orig_device)
        else:
            return mapped_output


class MappedModel(nn.Module):
    def __init__(self, model, map):
        super().__init__()
        self.model = model
        self.map = map

    def forward(self, image, return_probs=False):
        output = self.map(self.model(image), return_probs)
        return output.float().to(image.device) if 'mps' in str(image.device) else output


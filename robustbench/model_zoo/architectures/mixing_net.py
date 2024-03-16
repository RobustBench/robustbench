import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Union, List
from torch import Tensor


class PreActBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False)
            )

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out)
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out += shortcut
        return out


class MixingNetV3(nn.Module):
    def __init__(self, forward_settings, nmodels=2):
        super().__init__()
        #print("Initializing MixingNet V3.")
        self.nmodels = nmodels

        self.ind_planes = [None for _ in range(self.nmodels)]
        for ind, in_plane in enumerate(forward_settings["in_planes"]):
            if in_plane == 64:  # Number of channels of each layer
                self.ind_planes[ind] = (64, 128, 256, 512)
            elif in_plane == 160:
                self.ind_planes[ind] = (160, 320, 512, 768)
            elif in_plane == 256:
                self.ind_planes[ind] = (256, 256, 384, 512)
            elif in_plane == 512:
                self.ind_planes[ind] = (512, 256, 384, 512)
            else:
                raise ValueError("Unknown in_plane.")
        self.planes = np.array(self.ind_planes).sum(axis=0)

        self.in_planes = self.planes[0]
        self.layers = nn.ModuleList([self._make_layer(pl, stride=2) for pl in self.planes[1:]])

        self.global_avg_pool = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(self.planes[-1], 1, bias=False)

    def _make_layer(self, planes, stride, num_blocks=2):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(PreActBlock(self.in_planes, planes, stride))
            self.in_planes = planes * PreActBlock.expansion
        return nn.Sequential(*layers)

    def forward(self, feats1, feats2):
        feats1, feats2 = torch.cat(feats1, dim=1), torch.cat(feats2, dim=1)
        feats1 = self.layers[0](feats1)

        assert torch.all(feats1.shape == feats2.shape)
        feats = feats1 + feats2
        for l in range(1, len(self.layers)):
            feats = self.layers[l](feats)

        feats = self.global_avg_pool(feats).reshape(-1, self.planes[-1])
        feats = self.linear(feats)
        return feats


class MixingNetV4(MixingNetV3):
    def __init__(self, forward_settings, nmodels=2):
        super().__init__(forward_settings, nmodels)
        #print("Initializing MixingNet V4.")

        # Add 1x1 conv to reduce the number of channels
        in_1x1 = (self.ind_planes[0][0] + self.ind_planes[1][0]) * 2
        out_1x1 = self.ind_planes[0][1] + self.ind_planes[1][1]
        self.conv1x1 = nn.Conv2d(in_1x1, out_1x1, kernel_size=1, stride=1, bias=False)

    def forward(self, feats1, feats2):
        feats1, feats2 = torch.cat(feats1, dim=1), torch.cat(feats2, dim=1)
        feats = self.layers[0](feats1) + self.conv1x1(feats2)
        for l in range(1, len(self.layers)):
            feats = self.layers[l](feats)

        feats = self.global_avg_pool(feats).reshape(-1, self.planes[-1])
        feats = self.linear(feats)
        return feats


class NonLinMixedClassifier(nn.Module):
    def __init__(self, std_model: nn.Module, rob_model: nn.Module, forward_settings: dict):
        """ Initialize the MixedNUTS classifier.

        Args:
            std_model (nn.Module): The standard base classifier (can be non-robust).
            rob_model (nn.Module): The robust base classifier.
            forward_settings (dict): A dictionary containing the following forward settings:
                - use_nonlin_for_grad (bool):
                    If True, use the robust base model logit nonlinearity for gradient.
                    If False, (partially) bypass the nonlinearity for
                    better gradient flow and more effective attack.
                - std_map (nn.Module):
                    The mapping function for the standard base model logits.
                - rob_map (nn.Module):
                    The mapping function for the robust base model logits.
                - alpha (float or Tensor):
                    The mixing weight between the two base classifiers.
                    If a float, the MixedNUTS output has shape (n, num_classes).
                    If a Tensor with shape (m1, m2, ..., md), the MixedNUTS output has shape
                    (n, num_classes, m1, ..., md). I.e., an outer product is performed.
                - alpha_diffable (float or Tensor):
                    The mixing weight between the two base classifiers for differentiable output.
                    The shape-dependent behavior is the same as alpha.
        """
        super().__init__()
        self.std_model, self.rob_model = std_model, rob_model

        # Freeze models and set to eval mode
        for model, name in zip([self.std_model, self.rob_model], ["STD", "ROB"]):
            model.eval()
            #for param in model.parameters():
            #    assert param.requires_grad == False
            print(f"The {name} classifier has "
                  f"{sum(p.numel() for p in model.parameters())} parameters. "
                  f"{sum(p.numel() for p in model.parameters() if p.requires_grad)} "
                  "parameters are trainable.")

        # alpha value (mixing balance)
        self.set_alpha_value(forward_settings["alpha"], forward_settings["alpha_diffable"])
        self.use_nonlin_for_grad = forward_settings.get("use_nonlin_for_grad", False)
        print(f"{'Using' if self.use_nonlin_for_grad else 'Bypassing'} "
              "robust base model nonlinear transformation for gradient calculations.")

        # Base model logit transformations
        self.std_map, self.rob_map = forward_settings['std_map'], forward_settings['rob_map']

        # Enable autocast if specified
        self.enable_autocast = forward_settings.get("enable_autocast", False)
        print(f"{'Enabling' if self.enable_autocast else 'Disabling'} autocast.")

    def set_alpha_value(
        self, alpha: Union[float, int, Tensor, List],
        alpha_diffable: Union[float, int, Tensor, List], verbose=True
    ):
        """ Set the alpha value for the mixed classifier. """
        # Set alpha
        self._alpha = nn.parameter.Parameter(torch.tensor(alpha).float(), requires_grad=False)
        assert self._alpha.min() >= 0 and self._alpha.max() <= 1, \
            "The range of alpha should be [0, 1]."
        if verbose:
            print(f"Using alpha={alpha}.")
        if torch.numel(self._alpha) == 1 and self._alpha.item() == 0:
            print("Using the STD network only.")
        elif torch.numel(self._alpha) == 1 and self._alpha.item() == 1:
            print("Using the ROB network only.")

        # Set alpha_diffable
        self._alpha_diffable = nn.parameter.Parameter(
            torch.tensor(alpha_diffable).float(), requires_grad=False
        )
        assert self._alpha_diffable.min() >= 0 and self._alpha_diffable.max() <= 1, \
            "The range of alpha_diffable should be [0, 1]."
        if verbose and not torch.allclose(self._alpha_diffable, self._alpha, rtol=1e-8):
            print(f"Using alpha_diffable={alpha_diffable}.")
        if torch.numel(self._alpha_diffable) == 1 and self._alpha_diffable.item() == 0:
            print("Using the STD network only for differentiable output.")
        elif torch.numel(self._alpha_diffable) == 1 and self._alpha_diffable.item() == 1:
            print("Using the ROB network only for differentiable output.")

        assert self._alpha.shape == self._alpha_diffable.shape, \
            "alpha and alpha_diffable must have the same shape."

    def forward(self, images: Tensor, return_probs: bool = False, return_all: bool = False):
        """ Forward pass of the mixed classifier.

        Args:
            images (Tensor):
                Input images with size (n, c, h, w).
            return_probs (bool, optional):
                If True, skip and log and return the mixed probabilities.
                Otherwise, return MixedNUTS's logits. Defaults to False.
            return_all (bool, optional):
                If True, return the mixed probs/logits, the differentiable mixed probs/logits
                used for adversarial attack, and the alpha values.
                Otherwise, only return the mixed probs/logits for compatibility with
                existing models. Defaults to False.

        Returns:
            Tensor or tuple: Return values. See args for details.
        """
        assert not self.std_model.training, "The accurate base classifier should be in eval mode."
        assert not self.rob_model.training, "The robust base classifier should be in eval mode."
        return_device = images.device
        enable_autocast = self.enable_autocast and torch.cuda.is_available()
        raw_ratio = .9  # The ratio of the raw robust base model output to use for gradient

        # Base classifier forward passes
        # Accurate base classifier only
        if torch.numel(self._alpha) == 1 and self._alpha.item() == 0:
            with torch.cuda.amp.autocast(enabled=enable_autocast):
                logits_std = self.std_model(images)
            raw_std = logits_std.softmax(dim=1) if return_probs else logits_std
            mapped_std = self.std_map(logits_std, return_probs=return_probs).float()
            assert not mapped_std.isnan().any()
            alpha = torch.zeros((logits_std.shape[0],)).to(logits_std.device)
            return (mapped_std, raw_std, alpha) if return_all else mapped_std

        # Robust base classifier only
        elif torch.numel(self._alpha) == 1 and self._alpha.item() == 1:
            with torch.cuda.amp.autocast(enabled=enable_autocast):
                logits_rob = self.rob_model(images)
            raw_rob = logits_rob.softmax(dim=1) if return_probs else logits_rob
            mapped_rob = self.rob_map(logits_rob, return_probs=return_probs).float()
            assert not mapped_rob.isnan().any()
            alpha = torch.ones((logits_rob.shape[0],)).to(logits_rob.device)

            if self.use_nonlin_for_grad:
                grad_rob = raw_rob * raw_ratio + mapped_rob * (1 - raw_ratio)
            else:
                grad_rob = mapped_rob
            return (mapped_rob, grad_rob, alpha) if return_all else mapped_rob

        # General case -- use both models
        with torch.cuda.amp.autocast(enabled=enable_autocast):
            logits_std, logits_rob = self.std_model(images), self.rob_model(images)
        assert logits_std.device == logits_rob.device

        # Apply nonlinear logit transformations and convert to probabilities
        mapped_std = self.std_map(logits_std, return_probs=True)
        mapped_rob = self.rob_map(logits_rob, return_probs=True)

        alpha = self._alpha.to(mapped_rob.device)
        alphas_diffable = self._alpha_diffable.to(mapped_rob.device)

        # Mix the output probabilities of the two base classifiers
        mixed_probs = outer_prod((1 - alpha), mapped_std) + outer_prod(alpha, mapped_rob)
        # Log is the inverse of the softmax
        mixed_logits = torch.log(mixed_probs)
        assert not mixed_logits.isnan().any()

        if return_all:
            # Return mixed probs/logits, differentiable mixed probs/logits, and alphas

            if self.use_nonlin_for_grad:
                # Use the robust base model nonlinearity for gradient
                mixed_probs_diffable = outer_prod(alpha, mapped_rob) + \
                    outer_prod((1 - alpha), logits_std.softmax(dim=1).to(mapped_std.device))
                mixed_logits_diffable = torch.log(mixed_probs_diffable)
            else:
                # Disable the robust base model nonlinearity for gradient (usually stronger)
                probs_std = logits_std.softmax(dim=1).to(mapped_std.device)
                probs_rob = logits_rob.softmax(dim=1).to(mapped_rob.device)
                mixed_probs_diffable = \
                    outer_prod((1 - alphas_diffable), probs_std) + \
                    outer_prod(alphas_diffable * raw_ratio, probs_rob) + \
                    outer_prod(alphas_diffable * (1 - raw_ratio), mapped_rob)
                mixed_logits_diffable = torch.log(mixed_probs_diffable)

            assert not mixed_logits_diffable.isnan().any()
            if return_probs:
                return (
                    mixed_probs.float().to(return_device),
                    mixed_probs_diffable.float().to(return_device),
                    alpha.reshape(-1).to(return_device)
                )
            else:
                return (
                    mixed_logits.float().to(return_device),
                    mixed_logits_diffable.float().to(return_device),
                    alpha.reshape(-1).to(return_device)
                )

        else:
            # Only return the mixed probs/logits
            return mixed_probs.float().to(return_device) if return_probs \
                else mixed_logits.float().to(return_device)


def outer_prod(tensor1, tensor2):
    """ Outer product of two tensors of arbitrary shapes. """
    # Get the number of dimensions for each tensor
    dims1 = len(tensor1.shape)
    dims2 = len(tensor2.shape)

    # Construct the einsum string
    # For tensor1, we use 'abcd...' and for tensor2, we use 'efgh...'
    einsum_str = (
        ''.join(chr(i + 97) for i in range(dims1)) + ',' +
        ''.join(chr(i + 97 + dims1) for i in range(dims2)) + '->' +
        ''.join(chr(i + 97) for i in range(dims1 + dims2))
    )

    # Compute the outer product using einsum
    return torch.einsum(einsum_str, tensor1, tensor2)


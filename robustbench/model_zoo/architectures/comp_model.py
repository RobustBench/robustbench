import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import List
import numpy as np
import yaml
import os
import timm

from robustbench.model_zoo.architectures.mixing_net import MixingNetV3, MixingNetV4, \
    NonLinMixedClassifier
from robustbench.model_zoo.architectures import bit_rn, dm_rn, bit_rn_v2, convnext_v2
from robustbench.model_zoo.architectures.output_maps import HardMaxMap, LNClampPowerScaleMap
from robustbench.model_zoo.architectures.robustarch_wide_resnet import get_model as get_robustarch_model
from robustbench.model_zoo.architectures.dm_wide_resnet import CIFAR100_MEAN, CIFAR100_STD, \
    DMWideResNet
from robustbench.model_zoo.architectures.utils_architectures import normalize_model


class CompositeModel(nn.Module):
    """Here, we use the terms "mixing network" and "policy network" interchangeably.
    """
    enable_autocast = False
    sigmoid = nn.Sigmoid()
    softmax = nn.Softmax(dim=1)
    gamma = -np.inf

    def __init__(self, forward_settings):
        super().__init__()

        num_classes = forward_settings['num_classes']
        std_model_type = forward_settings["std_model_type"]
        model_name = "BiT-M-R50x1" if std_model_type == 'rn50' else "BiT-M-R152x2"
        rob_model_type = forward_settings["rob_model_type"]

        self.models = nn.ModuleList()

        self.models.append(
            bit_rn.KNOWN_MODELS[model_name](head_size=num_classes, zero_head=False))
        self.models.append(
            dm_rn.WideResNet(
                num_classes=num_classes,
                activation_fn=nn.SiLU if 'silu' in rob_model_type else dm_rn.Swish,
                depth=70,
                width=16,
                mean=dm_rn.CIFAR100_MEAN if num_classes == 100 else dm_rn.CIFAR10_MEAN,
                std=dm_rn.CIFAR100_STD if num_classes == 100 else dm_rn.CIFAR10_STD,
                ))

        for model in self.models:
            model.eval()
            for param in model.parameters():
                param.requires_grad = False
            #for param in model.parameters():
            #    assert param.requires_grad == False

        for model, typ in zip(self.models, ["STD", "ROB"]):
            print(f"The {typ} classifier has "
                  f"{sum(p.numel() for p in model.parameters())} parameters. "
                  f"{sum(p.numel() for p in model.parameters() if p.requires_grad)} "
                  "parameters are trainable.")

        self.policy_graph = forward_settings["policy_graph"]
        self.pn_version = forward_settings["pn_version"]
        if self.pn_version == 3:
            self.policy_net = MixingNetV3(forward_settings)
        elif self.pn_version == 4:
            self.policy_net = MixingNetV4(forward_settings)
        else:
            raise "Unsupported mixing network version."

        #self.policy_net = (
        #    self.policy_net.cuda() if forward_settings["parallel"] == 0 else self.policy_net)
        print("The mixing network has "
              f"{sum(p.numel() for p in self.policy_net.parameters())} parameters. "
              f"{sum(p.numel() for p in self.policy_net.parameters() if p.requires_grad)}"
              " parameters are trainable.\n")

        # Batch norm layer for the mixing network (act on gamma)
        self.bn = nn.BatchNorm1d(num_features=1, affine=False, momentum=0.01)
        self.softmax = nn.Softmax(dim=1)
        self.resize = forward_settings["std_model_type"] in ["rn50", "rn152"]

        # Set gamma and the use_policy flag
        self.use_policy = forward_settings["use_policy"]
        self.set_gamma_value(forward_settings["gamma"])  # Only useful without policy network
        self.scale_alpha = not self.training

        # Gamma and alpha scale and bias
        self.gamma_scale = nn.parameter.Parameter(torch.tensor(1.), requires_grad=False)  
        self.gamma_bias = nn.parameter.Parameter(torch.tensor(0.), requires_grad=False)
        self.alpha_scale = nn.parameter.Parameter(torch.tensor(1.), requires_grad=False)
        self.alpha_bias = nn.parameter.Parameter(torch.tensor(0.), requires_grad=False)
        self.std_scale = nn.parameter.Parameter(torch.tensor(1.), requires_grad=False)
        self.rob_scale = nn.parameter.Parameter(torch.tensor(1.), requires_grad=False)

        if "alpha_scale" in forward_settings.keys() and "alpha_bias" in forward_settings.keys():
            self.set_alpha_scale_bias(forward_settings["alpha_scale"], forward_settings["alpha_bias"])
        if "std_scale" in forward_settings.keys() and "rob_scale" in forward_settings.keys():
            self.set_base_model_scale(forward_settings["std_scale"], forward_settings["rob_scale"])

    def set_gamma_value(self, gamma):
        if self.use_policy:
            self.gamma = gamma
            print(f"gamma has been set to {self.gamma}, "
                  "but the mixing network is active so the change is not effective.")
        else:
            self.gamma = gamma
            print(f"Using fixed gamma={self.gamma}. No mixing network.")
            if self.gamma == -np.inf:
                print("Using the STD network only.")
            elif self.gamma == np.inf:
                print("Using the ROB network only.")

    def train(self, mode: bool=True, scale_alpha: bool=None):
        """Sets the mixing network and the BN in training mode. Overloads the train method of nn.Module.
        Args:
            mode (bool):        Whether to set training mode (``True``) or evaluation mode (``False``). 
                                Default: ``True``.
            scale_alpha (bool): Whether to scale alpha produced by the mixing network. 
                                If ``None``, then scale alpha iff in eval mode. Default: ``None``.
        """
        if not isinstance(mode, bool):
            raise ValueError("Training mode is expected to be boolean")

        self.training = mode
        self.policy_net.train(mode)
        self.bn.train(mode)

        if scale_alpha is None:
            scale_alpha = not mode  # Default setting is to scale gamma iff in evaluation mode.
        self.scale_alpha = scale_alpha
        return self

    def eval(self, scale_alpha: bool=None):
        return self.train(mode=False, scale_alpha=scale_alpha)

    def set_gamma_scale_bias(self, gamma_scale, gamma_bias):
        device = self.gamma_bias.device
        self.gamma_bias = nn.parameter.Parameter(
            torch.tensor(gamma_bias, device=device).float(), requires_grad=False)
        print(f"The mixing network's gamma mean is set to {self.gamma_bias.item()}.")
        self.gamma_scale = nn.parameter.Parameter(
            torch.tensor(gamma_scale, device=device).float(), requires_grad=False)
        print(f"The mixing network's gamma standard deviation is set to {self.gamma_scale.item()}.")

    def set_alpha_scale_bias(self, alpha_scale, alpha_bias):
        assert alpha_bias >= 0, "The range of alpha cannot be negative."
        assert alpha_scale + alpha_bias <= 1, "The range of alpha cannot exceed 1."
        device = self.alpha_bias.device
        self.alpha_bias = nn.parameter.Parameter(
            torch.tensor(alpha_bias, device=device).float(), requires_grad=False)
        self.alpha_scale = nn.parameter.Parameter(
            torch.tensor(alpha_scale, device=device).float(), requires_grad=False)
        print("The range of alpha during evaluation is set to "
              f"({self.alpha_bias.item()}, {(self.alpha_bias + self.alpha_scale).item()}).")

    def set_base_model_scale(self, std_scale, rob_scale):
        device = self.std_scale.device
        assert std_scale > 0 and rob_scale > 0, \
            "The logit output scale of the base models must be positive."
        self.std_scale = nn.parameter.Parameter(
            torch.tensor(std_scale, device=device).float(), requires_grad=False)
        print(f"The logit output scale of the STD network is set to {self.std_scale.item()}.")
        self.rob_scale = nn.parameter.Parameter(
            torch.tensor(rob_scale, device=device).float(), requires_grad=False)
        print(f"The logit output scale of the ROB network is set to {self.rob_scale.item()}.")

    def do_checks(self, images):
        if self.policy_graph and not self.use_policy:
            raise ValueError('policy_graph cannot be created without the mixing network.')
        for model in self.models:
            assert not model.training

        if hasattr(self.models[0], "root") and (
            self.models[0].root.conv.weight.device != images.device):
            print(self.models[0].root.conv.weight.device, 
                  self.models[1].logits.weight.device, 
                  self.policy_net.linear.weight.device, 
                  self.bn.running_mean.device, images.device)
            raise ValueError("Device mismatch!")

    def forward(self, images):
        self.do_checks(images)

        # The STD model requires resized images
        images_resized = F.interpolate(images, size=(128, 128), mode='bilinear') if (
            self.resize and (self.gamma != np.inf or self.use_policy)) else images

        # Forward passes
        with torch.cuda.amp.autocast(enabled=self.enable_autocast):

            # Single model special cases
            if (self.gamma == -np.inf and not self.use_policy) or (
                self.alpha_scale == 0. and self.alpha_bias == 0.):  # STD Model only
                out, _ = self.models[0](images_resized)
                gamma = torch.tensor(-np.inf) * torch.ones((out.shape[0],)).to(out.device)
                return out * self.std_scale, gamma

            elif (self.gamma == np.inf and not self.use_policy) or self.alpha_bias == 1.:
                out, _ = self.models[1](images)  # ROB Model only
                gamma = torch.tensor(np.inf) * torch.ones((out.shape[0],)).to(out.device)
                return out * self.rob_scale, gamma

            # General case -- use both models
            out_data_std, interm_std = self.models[0](images_resized)
            out_data_rob, interm_rob = self.models[1](images)

            if self.use_policy and self.alpha_scale != 0:  # Use the mixing network
                if self.policy_graph:
                    gammas = self.policy_net(
                        [interm_std[0], interm_rob[0]], [interm_std[1], interm_rob[1]])
                else:
                    gammas = self.policy_net(
                        [interm_std[0].detach().clone(), interm_rob[0].detach().clone()],
                        [interm_std[1].detach().clone(), interm_rob[1].detach().clone()])

                # Clamp gammas during training so that the output BN works the best
                if self.training:
                    amean, astd = gammas.mean().item(), gammas.std().item()
                    gammas = torch.clamp(gammas, min=(-.6) * astd + amean, max=.6 * astd + amean)

                # Apply BN and reparameterize
                gammas = self.bn(gammas) * self.gamma_scale + self.gamma_bias
                # print(gammas.mean().item(), gammas.median().item(), (gammas>=0).float().mean().item())
                alphas = self.sigmoid(gammas)

                # If scale_alpha is specified (default option in eval mode), shrink the range of alphas.
                if self.scale_alpha:
                    alphas = alphas * self.alpha_scale + self.alpha_bias

            elif self.use_policy:  # alpha_scale is 0, so use the bias
                alphas = self.alpha_bias * torch.ones(
                    (out_data_std.shape[0], 1), device=out_data_std.device)

            else:  # Use fixed gamma
                gammas = self.gamma * torch.ones(
                    (out_data_std.shape[0], 1), device=out_data_std.device)
                alphas = self.sigmoid(gammas)

        out_data = torch.log(  # Log is the inverse of the softmax
            (1 - alphas) * self.softmax(out_data_std * self.std_scale) + 
            alphas * self.softmax(out_data_rob * self.rob_scale))

        return out_data, gammas.reshape(-1)


class CompositeModelWrapper(nn.Module):
    """
    A wrapper for the composite model that only returns the first output.
    This is used for compatibility with RobustBench.
    """
    def __init__(self, comp_model, parallel=True):
        super().__init__()

        self.comp_model = comp_model
        if parallel:
            print("Parallelizing the entire composite model.")
            self.comp_model = nn.DataParallel(self.comp_model)
            self._comp_model = self.comp_model.module  # This is the unparallelized model
        else:
            self._comp_model = self.comp_model
        print("")  # Print a blank line

    def forward(self, images):
        return self.comp_model(images)[0]

    def train(self, mode: bool=True, scale_alpha: bool=None):
        self.training = mode
        self._comp_model.train(mode=mode, scale_alpha=scale_alpha)
        return self

    def eval(self, scale_alpha: bool=None):
        return self.train(mode=False, scale_alpha=scale_alpha)


def get_composite_model(model_name, dataset='cifar100'):

    forward_settings = {
        "std_model_type": 'rn152',
        "rob_model_type": 'wrn7016_silu' if model_name == 'edm' else 'wrn7016',
        "in_planes": (512, 256),
        "gamma": 2.5 if model_name == 'edm' else 1.75,  # Overriden by the mixing network.
        "use_policy": True,  # False if no mixing network
        "policy_graph": True,  # False if no mixing network
        "pn_version": 4,
        "parallel": False,
        "num_classes": 100 if dataset == 'cifar100' else 10,
    }

    comp_model = CompositeModel(forward_settings)

    # Set output mean and std div. Only applies when the policy network is active.
    if dataset == 'cifar100' and model_name == "edm":  # EDM
        comp_model.set_gamma_scale_bias(gamma_scale=2., gamma_bias=1.3)
        comp_model.set_alpha_scale_bias(alpha_scale=.15, alpha_bias=.84)
        # Set base model logit scale
        comp_model.set_base_model_scale(std_scale=2., rob_scale=1.)
    elif dataset == 'cifar100' and model_name == "trades":  # TRADES
        comp_model.set_gamma_scale_bias(gamma_scale=2., gamma_bias=1.)
        comp_model.set_alpha_scale_bias(alpha_scale=.1, alpha_bias=.815)
        # Set base model logit scale
        comp_model.set_base_model_scale(std_scale=2., rob_scale=1.)
    elif dataset == 'cifar10' and model_name == 'edm':
        comp_model.set_gamma_value(gamma=3.)
        comp_model.set_gamma_scale_bias(gamma_scale=2., gamma_bias=1.05)
        comp_model.set_alpha_scale_bias(alpha_scale=.04, alpha_bias=.96)
        comp_model.set_base_model_scale(std_scale=1.2, rob_scale=.3)
    else:
        raise ValueError(f"Unknown model name: {model_name}.")

    return CompositeModelWrapper(comp_model, parallel=False)


# Adapted from
# https://github.com/Bai-YT/MixedNUTS/blob/main/robustbench_impl/utils/model_utils.py.
def get_nonlin_mixed_classifier(dataset_name):
    """ This function is used to build the mixed classifier for RobustBench. """

    std_model_arch, rob_model_name, num_classes = {
        'cifar10': ('rn152', 'Peng2023Robust', 10),
        'cifar100': ('rn152', 'Wang2023Better_WRN-70-16', 100),
        'imagenet': ('convnext_v2-l_224', 'Liu2023Comprehensive_Swin-L', 1000),
    }[dataset_name]

    # Default transformation hyperparameters
    config_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'optimal_spca.yaml')
    with open(config_file, 'r') as file:
        dflt_dic = yaml.safe_load(file)[dataset_name][rob_model_name]

    beta_diffable = dflt_dic['none']['default_beta']
    alpha_diffable = dflt_dic['none'][beta_diffable]['alpha']
    std_map = HardMaxMap()
    beta = dflt_dic['gelu']['default_beta']
    nonlin_spca = dflt_dic['gelu'][beta]
    rob_map = LNClampPowerScaleMap(
        scale=nonlin_spca['s'], power=nonlin_spca['p'], clamp_bias=nonlin_spca['c'], ln_k=250
    )
    alpha = nonlin_spca['alpha']
    forward_settings = {
        "std_model_arch": std_model_arch,
        "rob_model_name": rob_model_name,
        "std_map": std_map,
        "rob_map": rob_map,
        "alpha": alpha,
        "alpha_diffable": alpha_diffable,
        "parallel": False
    }

    # Load robust base model
    if rob_model_name == 'Peng2023Robust':
        rob_model = get_robustarch_model('ra_wrn70_16')
    elif rob_model_name == 'Wang2023Better_WRN-70-16':
        rob_model = DMWideResNet(
            num_classes=100, depth=70, width=16, activation_fn=nn.SiLU,
            mean=CIFAR100_MEAN, std=CIFAR100_STD)
    elif rob_model_name == 'Liu2023Comprehensive_Swin-L':
        mu = (0.485, 0.456, 0.406)
        sigma = (0.229, 0.224, 0.225)
        rob_model = normalize_model(timm.create_model(
            'swin_large_patch4_window7_224', pretrained=False), mu, sigma)

    # Load standard base model
    if std_model_arch == 'convnext_v2-l_224':
        # Load ConvNext V2 model
        assert num_classes == 1000
        std_model = convnext_v2.convnextv2_large(num_classes=num_classes)
    elif std_model_arch == 'rn152':
        # Load BiT ResNet model
        std_model = bit_rn_v2.KNOWN_MODELS["BiT-M-R152x2"](
            head_size=num_classes, zero_head=False, return_features=False)
    else:
        raise ValueError(f"Unknown standard model architecture: {std_model_arch}")

    # Mixed classifier
    mix_model = NonLinMixedClassifier(std_model, rob_model, forward_settings)

    return mix_model


if __name__ == '__main__':

    #model = get_composite_model('edm')
    model = get_nonlin_mixed_classifier('imagenet')
    device = 'cuda:6'
    model.to(device)
    x = torch.rand([10, 3, 224, 224], device=device)
    with torch.no_grad():
        print(model(x).shape)


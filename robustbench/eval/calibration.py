"""
The temperature scaling implementation is based on:
https://github.com/gpleiss/temperature_scaling
"""

import torch
import math
import json
import numpy as np
from pathlib import Path

from robustbench.data import load_clean_dataset
from robustbench.model_zoo.models import model_dicts
from torch import nn, optim
from torch.nn import functional as F


class ECELoss(nn.Module):
    """
    Calculates the Expected Calibration Error of a model.
    (This isn't necessary for temperature scaling, just a cool metric).
    The input to this loss is the logits of a model, NOT the softmax scores.
    This divides the confidence outputs into equally-sized interval bins.
    In each bin, we compute the confidence gap:
    bin_gap = | avg_confidence_in_bin - accuracy_in_bin |
    We then return a weighted average of the gaps, based on the number
    of samples in each bin
    See: Naeini, Mahdi Pakdaman, Gregory F. Cooper, and Milos Hauskrecht.
    "Obtaining Well Calibrated Probabilities Using Bayesian Binning." AAAI.
    2015.
    """
    def __init__(self, n_bins=15):
        """
        n_bins (int): number of confidence interval bins
        """
        super(ECELoss, self).__init__()
        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        self.bin_lowers = bin_boundaries[:-1]
        self.bin_uppers = bin_boundaries[1:]

    def forward(self, logits, labels):
        softmaxes = F.softmax(logits, dim=1)
        confidences, predictions = torch.max(softmaxes, 1)
        accuracies = predictions.eq(labels)

        ece = torch.zeros(1, device=logits.device)
        for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
            # Calculated |confidence - accuracy| in each bin
            in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
            prop_in_bin = in_bin.float().mean()
            if prop_in_bin.item() > 0:
                accuracy_in_bin = accuracies[in_bin].float().mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

        return ece


def temperature_calibration_lbfgs(logits, y_test):
    """
    Iterative optimization for a one-dimensional non-convex optimization doesn't seem like a good idea.
    Particularly that here the cross-entropy loss is optimized instead of directly ECE.
    """
    def temperature_scale(logits, temperature):
        """
        Perform temperature scaling on logits.
        """
        # Expand temperature to match the size of logits
        temperature = temperature.unsqueeze(1).expand(logits.size(0), logits.size(1))
        return logits / temperature

    temperature = (torch.ones(1, device='cuda') * 1.5).requires_grad_()
    # Next: optimize the temperature w.r.t. NLL
    optimizer = optim.LBFGS([temperature], lr=0.01, max_iter=50)

    def eval():
        criterion = nn.CrossEntropyLoss().cuda()
        loss = criterion(temperature_scale(logits, temperature), y_test)
        loss.backward()
        return loss

    optimizer.step(eval)
    return temperature.item()


def temperature_calibration(logits, y_test, temperature_grid_step=0.001):
    """
    Temperature calibration via a grid search over ECE (and not cross-entropy loss).
    """
    temperature_grid = np.arange(0.001, 1.0, temperature_grid_step)
    temperature_grid = np.concatenate([temperature_grid, 1 / temperature_grid])
    ece_criterion = ECELoss().cuda()

    ece_opt, temperature_opt = ece_criterion(logits, y_test), 1.0  # init as temperature=1.0
    for temperature in temperature_grid:
        ece = ece_criterion(logits / temperature, y_test)
        if ece < ece_opt:
            ece_opt, temperature_opt = ece, temperature

    return temperature_opt


def get_logits(model, x, batch_size, device):
    """
    TODO: this function has to be moved to utils.py afterwards. but so far it's easy to keep ALL my modifications
          in a single file.
    """
    n_batches = math.ceil(x.shape[0] / batch_size)
    logits_all = []
    with torch.no_grad():
        for counter in range(n_batches):
            x_curr = x[counter * batch_size:(counter + 1) * batch_size].to(device)
            logits = model(x_curr)
            logits_all.append(logits)
    return torch.cat(logits_all)


def compute_calibration():
    # TODO: unify the interface with the benchmark function (as in the Lipschitz function of Edoardo)
    from robustbench.utils import load_model
    from robustbench.utils import clean_accuracy

    # TODO: produce plots based on the printing outputs
    # TODO: also produce "reliability diagrams" for a few selected models:
    #  https://gist.github.com/gpleiss/0b17bc4bd118b49050056cfcd5446c71
    # TODO: also read in detail the arguments of Guo et al. https://arxiv.org/pdf/1706.04599.pdf and provide a discussion
    #  based on our findings.
    n_ex = 1000  # TODO: increase to 10k at the end
    batch_size = 256
    device = torch.device("cuda")

    ece_criterion = ECELoss().cuda()
    dataset_list, threat_model_list, acc_list, rob_acc_list, extra_data_list, arch_list = [], [], [], [], [], []  # from jsons
    accs_reproduced_list, ece_list, ece_t_list, t_opt_list = [], [], [], []  # new stats
    for dataset, dataset_dict in model_dicts.items():
        x_test, y_test = load_clean_dataset(dataset, n_ex, data_dir='./data')

        for threat_model, threat_model_dict in dataset_dict.items():
            # if threat_model.value != 'corruptions': continue
            models = list(threat_model_dict.keys())

            for model_name in models:
                model_info_path = Path("model_info") / dataset.value / threat_model.value / f"{model_name}.json"
                with open(model_info_path) as model_info:
                    json_dict = json.load(model_info)

                model = load_model(model_name, './models',
                                   dataset, threat_model).to(device)
                acc = clean_accuracy(model, x_test, y_test, batch_size=batch_size, device=device)
                if acc < 0.5:
                    print('----- Warning: acc < 0.5! Was the model restored correctly? ------')
                logits = get_logits(model, x_test, batch_size, device)
                ece = ece_criterion(logits, y_test.to(device)).item()

                t_opt = temperature_calibration(logits, y_test.to(device))
                ece_calibrated = ece_criterion(logits / t_opt, y_test.to(device)).item()

                rob_acc_field = 'autoattack_acc' if threat_model.value in ('Linf', 'L2') else 'corruptions_acc'
                print('Dataset={}, threat_model={}, model={}: acc={:.1%}, rob_acc={:.1%}, ece={:.2%}, ece_t={:.2%} (t={:.3f}) ({} ex; acc={:.1%})'.format(
                    dataset.value, threat_model.value, model_name, float(json_dict['clean_acc'])/100, float(json_dict[rob_acc_field])/100,
                    ece, ece_calibrated, t_opt, n_ex, acc))

                dataset_list.append(dataset.value)
                threat_model_list.append(threat_model.value)
                acc_list.append(float(json_dict['clean_acc'])/100)
                rob_acc_list.append(float(json_dict[rob_acc_field])/100)
                extra_data_list.append(json_dict['additional_data'])
                arch_list.append(json_dict['architecture'])
                accs_reproduced_list.append(acc)
                ece_list.append(ece)
                ece_t_list.append(ece_calibrated)
                t_opt_list.append(t_opt)

            calibration_stats = {
                'n_ex': n_ex,
                'dataset': np.array(dataset_list), 'threat_model': np.array(threat_model_list), 'acc': np.array(acc_list),
                'rob_acc': np.array(rob_acc_list), 'extra_data': np.array(extra_data_list), 'arch': np.array(arch_list),
                'accs_reproduced': np.array(accs_reproduced_list), 'ece': np.array(ece_list), 'ece_t': np.array(ece_t_list),
                't_opt': np.array(t_opt_list),
            }
            np.save('model_info/calibration_stats.npy', calibration_stats)


if __name__ == '__main__':
    compute_calibration()


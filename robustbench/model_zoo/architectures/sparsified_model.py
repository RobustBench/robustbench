import torch
import torch.nn as nn
from timm.layers.activations import GELU

class MeanSparse_cifar_10(nn.Module):
    def __init__(self, in_planes, momentum=0.1):
        super(MeanSparse_cifar_10, self).__init__()

        self.register_buffer('running_mean', torch.zeros(in_planes))
        self.register_buffer('running_var', torch.zeros(in_planes))

        self.register_buffer('threshold', torch.tensor(0.0))

        self.register_buffer('flag_update_statistics', torch.tensor(0))
        self.register_buffer('batch_num', torch.tensor(0.0))

    def forward(self, input):
        
        if self.flag_update_statistics:
            self.running_mean += (torch.mean(input.detach().clone(), dim=(0, 2, 3))/self.batch_num)
            self.running_var += (torch.var(input.detach().clone(), dim=(0, 2, 3))/self.batch_num)

        bias = self.running_mean.view(1, self.running_mean.shape[0], 1, 1)
        crop = self.threshold * torch.sqrt(self.running_var).view(1, self.running_var.shape[0], 1, 1)

        diff = input - bias

        if self.threshold == 0:
            output = input
        else:
            output = torch.where(torch.abs(diff) < crop, bias*torch.ones_like(input), input)

        return output

def add_custom_layer_cifar_10(model, custom_layer_class, parent_path='', prev_features=None):
    for name, child in model.named_children():
        current_path = f"{parent_path}.{name}" if parent_path else name  # Build the current path

        if isinstance(child, nn.SiLU):
            modified_layer = nn.Sequential(custom_layer_class(prev_features), child)
            setattr(model, name, modified_layer)
        
        elif isinstance(child, nn.BatchNorm2d):
            prev_features = child.num_features

        add_custom_layer_cifar_10(child, custom_layer_class, current_path, prev_features)

def add_custom_layer_imagenet(model, custom_layer_class, parent_path='', prev_features=None):
    for name, child in model.named_children():
        current_path = f"{parent_path}.{name}" if parent_path else name  # Build the current path

        if name == 'drop_path':  # Extend this tuple with other activation types if needed
            if 'stages.0' in parent_path:
                modified_layer = nn.Sequential(child, custom_layer_class(192))
            elif 'stages.1' in parent_path:
                modified_layer = nn.Sequential(child, custom_layer_class(384))
            elif 'stages.2' in parent_path:
                modified_layer = nn.Sequential(child, custom_layer_class(768))
            elif 'stages.3' in parent_path:
                modified_layer = nn.Sequential(child, custom_layer_class(1536))

            setattr(model, name, modified_layer)

        elif isinstance(child, GELU):
            if 'stages.0' in parent_path:
                modified_layer = nn.Sequential(custom_layer_class(prev_features), child)
            elif 'stages.1' in parent_path:
                modified_layer = nn.Sequential(custom_layer_class(prev_features), child)
            elif 'stages.2' in parent_path:
                modified_layer = nn.Sequential(custom_layer_class(prev_features), child)
            elif 'stages.3' in parent_path:
                modified_layer = nn.Sequential(custom_layer_class(prev_features), child)
            setattr(model, name, modified_layer)
        
        elif isinstance(child, nn.Linear):
            prev_features = child.out_features

        add_custom_layer_imagenet(child, custom_layer_class, current_path, prev_features)

class MeanSparse_imagenet(nn.Module):
    def __init__(self, in_planes, momentum=0.1):
        super(MeanSparse_imagenet, self).__init__()

        self.register_buffer('momentum', torch.tensor(momentum))
        self.register_buffer('epsilon', torch.tensor(1.0e-10))

        self.register_buffer('running_mean', torch.zeros(in_planes))
        self.register_buffer('running_var', torch.zeros(in_planes))

        self.register_buffer('threshold', torch.tensor(0.0))
        # self.coefficient = nn.Parameter(torch.tensor(torch.inf), requires_grad=False)

        self.register_buffer('flag_update_statistics', torch.tensor(0))
        self.register_buffer('batch_num', torch.tensor(0.0))

    def forward(self, input):
        
        if input.shape[1] == self.running_mean.shape[0]:
            if self.flag_update_statistics:
                self.running_mean += (torch.mean(input.detach().clone(), dim=(0, 2, 3))/self.batch_num)
                self.running_var += (torch.var(input.detach().clone(), dim=(0, 2, 3))/self.batch_num)

            bias = self.running_mean.view(1, self.running_mean.shape[0], 1, 1)
            # interval = self.coefficient * torch.sqrt(self.running_var).view(1, self.running_var.shape[0], 1, 1)

            crop = self.threshold * torch.sqrt(self.running_var).view(1, self.running_var.shape[0], 1, 1)

            diff = input - bias

            if self.threshold == 0:
                output = input
            else:
                output = torch.where(torch.abs(diff) < crop, bias*torch.ones_like(input), input)
        
        else:
            if self.flag_update_statistics:
                self.running_mean += (torch.mean(input.detach().clone(), dim=(0, 1, 2))/self.batch_num)
                self.running_var += (torch.var(input.detach().clone(), dim=(0, 1, 2))/self.batch_num)

            bias = self.running_mean.view(1, 1, 1, self.running_mean.shape[0])
            # interval = self.coefficient * torch.sqrt(self.running_var).view(1, self.running_var.shape[0], 1, 1)

            crop = self.threshold * torch.sqrt(self.running_var).view(1, 1, 1, self.running_var.shape[0])

            diff = input - bias

            if self.threshold == 0:
                output = input
            else:
                output = torch.where(torch.abs(diff) < crop, bias*torch.ones_like(input), input)

        return output

def get_sparse_model(model, dataset='cifar-10'):
    if dataset == 'cifar-10':
        add_custom_layer_cifar_10(model, MeanSparse_cifar_10, parent_path='', prev_features=None)
    elif dataset == 'imagenet':
        add_custom_layer_imagenet(model, MeanSparse_imagenet, parent_path='', prev_features=None)
    
    return model 
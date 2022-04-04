"""
Model definitions to load `pth` files.
"""

import logging

import torch
from torch import nn
from URSABench import models
from URSABench.models.wideresnet import WideResNet_dropout

logger = logging.getLogger("URSABench")
logger.setLevel(logging.DEBUG)
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(levelname)s %(pathname)s:%(lineno)d] %(message)s")
    ch.setFormatter(formatter)
    logger.addHandler(ch)


class MLP(torch.nn.Module):
    def __init__(self, input_size, hidden_size):
        super(MLP, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.fc1 = torch.nn.Linear(self.input_size, self.hidden_size)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(self.hidden_size, 1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        hidden = self.fc1(x)
        relu = self.relu(hidden)
        output = self.fc2(relu)
        output = self.sigmoid(output)
        return output


class MLPEnsemble(torch.nn.Module):
    def __init__(self, input_size, hidden_size):
        super(MLPEnsemble, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.fc1_list = nn.ModuleList(
            [nn.Linear(self.input_size, self.hidden_size) for i in range(30)]
        )
        #         self.fc1 = torch.nn.Linear(self.input_size, self.hidden_size)
        self.relu_list = nn.ModuleList([torch.nn.ReLU() for i in range(30)])
        self.fc2_list = nn.ModuleList(
            [nn.Linear(self.hidden_size, 1) for i in range(30)]
        )
        self.sigmoid_list = nn.ModuleList([torch.nn.Sigmoid() for i in range(30)])

    def forward(self, x):
        hidden_list = [fc1(x) for fc1 in self.fc1_list]
        relu_list = [relu(hidden) for relu, hidden in zip(self.relu_list, hidden_list)]
        output_list = [fc2(relu) for fc2, relu in zip(self.fc2_list, relu_list)]
        output_list = [
            sigmoid(output) for sigmoid, output in zip(self.sigmoid_list, output_list)
        ]
        return torch.stack(output_list).mean(dim=0)


class MLPEnsemble2(torch.nn.Module):
    def __init__(self, input_size, hidden_size):
        super(MLPEnsemble2, self).__init__()
        self.module_list = nn.ModuleList(
            [MLP(input_size, hidden_size) for i in range(30)]
        )

    def forward(self, x):
        for mlp in self.module_list:
            for params in mlp.parameters():
                params.requires_grad = False
        output_list = [mlp(x) for mlp in self.module_list]
        return torch.stack(output_list).mean(dim=0)


wrn_cfg = getattr(models, "WideResNet28x10")


class WRNEnsemble2(torch.nn.Module):
    def __init__(self):
        super(WRNEnsemble2, self).__init__()
        self.module_list = [
            wrn_cfg.base(*wrn_cfg.args, **wrn_cfg.kwargs, num_classes=10)
            for i in range(2)
        ]

    def forward(self, x):
        for m in self.module_list:
            for params in m.parameters():
                params.requires_grad = False
        output_list = [model(x) for model in self.module_list]
        return torch.stack(output_list).mean(dim=0)


rn50_cfg = getattr(models, "INResNet50")


class ResNet50Ensemble2(torch.nn.Module):
    def __init__(self):
        super(ResNet50Ensemble2, self).__init__()
        self.module_list = [
            rn50_cfg.base(*rn50_cfg.args, **rn50_cfg.kwargs, num_classes=10)
            for i in range(3)
        ]

    def forward(self, x):
        for m in self.module_list:
            for params in m.parameters():
                params.requires_grad = False
        output_list = [model(x) for model in self.module_list]
        return torch.stack(output_list).mean(dim=0)


class WRNDEnsemble2(torch.nn.Module):
    def __init__(self):
        super(WRNDEnsemble2, self).__init__()
        self.module_list = [WideResNet_dropout(num_classes=10) for i in range(3)]

    def forward(self, x):
        for m in self.module_list:
            for params in m.parameters():
                params.requires_grad = False
        output_list = [model(x) for model in self.module_list]
        return torch.stack(output_list).mean(dim=0)

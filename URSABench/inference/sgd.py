from copy import deepcopy

import torch
import wandb
from torch.optim.lr_scheduler import CosineAnnealingLR

from URSABench.util import get_loss_criterion, reset_model
from . import optimSGHMC
from .inference_base import _Inference


# import pyvarinf
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class SGD(_Inference):

    def __init__(self, hyperparameters, model=None, train_loader=None, model_loss='multi_class_linear_output',
                 device=torch.device('cpu')):
        '''
        :param hyperparameters: Hyperparameters include {'lr', 'prior_std', 'num_samples'}
        :param model: Pytorch model to run SGLD on.
        :param train_loader: DataLoader for train data
        :param model_loss: Loss function to use for the model. (e.g.: 'multi_class_linear_output')
        :param device: Device on which model is present (e.g.: torch.device('cpu'))
        '''
        if hyperparameters == None:
            # Initialise as some default values
            hyperparameters = {'lr': 0.1, 'epochs':10, 'momentum': 0.9, 'weight_decay': 0.001}

        super(SGD, self).__init__(hyperparameters, model, train_loader, device)
        self.lr = hyperparameters['lr']
        self.num_samples = 1
        self.burn_in_epochs = hyperparameters['epochs']
        self.momentum = hyperparameters['momentum']
        self.model = model.to(device)

        self.train_loader = train_loader
        self.device = device
        self.dataset_size = len(train_loader.dataset)
        self.weight_decay = hyperparameters['weight_decay']
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr, momentum=self.momentum,
                                         weight_decay=self.weight_decay)
        self.loss_criterion = get_loss_criterion(loss=model_loss)
        self.burnt_in = False
        self.epochs_run = 0
        self.lr_final = self.lr / 100.
        self.optimizer_scheduler = CosineAnnealingLR(optimizer=self.optimizer, T_max=
        self.burn_in_epochs + self.num_samples, eta_min=self.lr_final)

    def update_hyp(self, hyperparameters):
        self.lr = hyperparameters['lr']
        self.num_samples = 1
        self.epochs = hyperparameters['epochs']
        self.momentum = hyperparameters['momentum']
        self.weight_decay = hyperparameters['weight_decay']
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr, momentum=self.momentum,
                                         weight_decay=self.weight_decay)
        self.model = reset_model(self.model).to(self.device)
        self.burnt_in = False
        self.epochs_run = 0
        self.lr_final = self.lr / 2
        self.optimizer_scheduler = CosineAnnealingLR(optimizer=self.optimizer, T_max=
        self.burn_in_epochs + self.num_samples, eta_min=self.lr_final)

    def sample_iterative(self, val_loader=None, debug_val_loss=False, wandb_debug=False):
        if issubclass(self.model.__class__, torch.nn.Module):
            if self.burnt_in is False:
                epochs = self.burn_in_epochs + 1
                self.burnt_in = True
            else:
                epochs = 0
            for epoch in range(epochs):
                self.model.train()
                total_epoch_train_loss = 0.
                for batch_idx, (batch_data, batch_labels) in enumerate(self.train_loader):
                    batch_data = batch_data.to(self.device)
                    batch_labels = batch_labels.to(self.device)
                    batch_data_logits = self.model(batch_data)
                    self.optimizer.zero_grad()
                    loss = self.loss_criterion(batch_data_logits, batch_labels)
                    loss.backward()
                    total_epoch_train_loss += loss.item() * len(batch_data)
                    self.optimizer.step()
                if debug_val_loss:
                    avg_val_loss = self.compute_val_loss(val_loader)
                    avg_train_loss = total_epoch_train_loss / self.dataset_size
                    metrics = {
                        'train_loss': avg_train_loss,
                        'val_loss': avg_val_loss
                    }
                    print(metrics)
                    if wandb_debug:
                        wandb.log(metrics)
                self.optimizer_scheduler.step()
            return self.model
        else:
            raise NotImplementedError

    def sample(self, num_samples=None, val_loader=None, debug_val_loss=False, wandb_debug=False):
        output_list = []
        if num_samples is None:
            num_samples = self.num_samples
        if issubclass(self.model.__class__, torch.nn.Module):
            for i in range(num_samples):
                output_list.append(self.sample_iterative(val_loader=val_loader, debug_val_loss=debug_val_loss,
                                                         wandb_debug=wandb_debug))
            return output_list
        else:
            raise NotImplementedError

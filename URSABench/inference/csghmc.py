from copy import deepcopy

import numpy as np
import torch
import wandb

from URSABench.util import get_loss_criterion, reset_model
from .inference_base import _Inference
from .optim_sghmc import optimSGHMC


# TODO: Add docstrings for classes below.
class cSGHMC(_Inference):
    def __init__(self, hyperparameters, model=None, train_loader=None, model_loss='multi_class_linear_output',
                 device=torch.device('cpu')):

        if hyperparameters == None:
            # Initialise as some default values
            hyperparameters = {'lr_0': 0.001000, 'prior_std': 10.1000, 'num_samples_per_cycle': 5, 'cycle_length': 20, 'burn_in_epochs': 5, 'num_cycles': 10, 'alpha': 1.,}

        super(cSGHMC, self).__init__(hyperparameters, model, train_loader, device)
        self.lr_0 = hyperparameters['lr_0']
        self.prior_std = hyperparameters['prior_std']
        self.num_samples_per_cycle = hyperparameters['num_samples_per_cycle']
        self.cycle_length = hyperparameters['cycle_length']
        self.alpha = hyperparameters['alpha']
        self.burn_in_epochs = hyperparameters['burn_in_epochs']
        self.num_cycles = hyperparameters['num_cycles']
        self.batch_size = train_loader.batch_size
        self.num_batch = len(train_loader.dataset) / self.batch_size + 1
        self.num_batch = max(1, self.num_batch)
        self.model_loss = model_loss
        self.model = model
        self.train_loader = train_loader
        self.device = device
        self.dataset_size = len(train_loader.dataset)
        self.optimizer = optimSGHMC(params=self.model.parameters(), lr=self.lr_0, momentum=1 - self.alpha,
                                    num_training_samples=self.dataset_size, weight_decay=1 / (self.prior_std ** 2))
        self.loss_criterion = get_loss_criterion(loss=model_loss)
        self.burnt_in = False
        self.epochs_run = 0
        self.total_epochs = self.cycle_length * self.num_cycles
        self.dataloader_batch_size = self.train_loader.batch_size
        self.total_iterations = self.total_epochs * self.num_batch

        assert ((self.cycle_length - self.burn_in_epochs - self.num_samples_per_cycle) > 0)

    def update_hyp(self, hyperparameters):
        self.lr_0 = hyperparameters['lr_0']
        self.prior_std = hyperparameters['prior_std']
        self.num_samples_per_cycle = hyperparameters['num_samples_per_cycle']
        self.cycle_length = hyperparameters['cycle_length']
        self.alpha = hyperparameters['alpha']
        self.burn_in_epochs = hyperparameters['burn_in_epochs']
        self.num_cycles = hyperparameters['num_cycles']
        self.model = reset_model(self.model)
        self.optimizer = optimSGHMC(params=self.model.parameters(), lr=self.lr_0, momentum=1 - self.alpha,
                                    num_training_samples=self.dataset_size, weight_decay=1 / (self.prior_std ** 2))
        self.burnt_in = False
        self.epochs_run = 0

        assert ((self.cycle_length - self.burn_in_epochs - self.num_samples_per_cycle) > 0)

    def _adjust_learning_rate(self, optimizer, epoch, batch_idx):
        rcounter = epoch * self.num_batch + batch_idx
        cos_inner = np.pi * (rcounter % (self.total_iterations // self.num_cycles))
        cos_inner /= self.total_iterations // self.num_cycles
        cos_out = np.cos(cos_inner) + 1
        lr = 0.5 * cos_out * self.lr_0
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        return lr

    def sample_iterative(self, val_loader=None, debug_val_loss=False, wandb_debug=False):
        if issubclass(self.model.__class__, torch.nn.Module):
            sample_collected = False
            while sample_collected is False:
                self.model.train()
                total_epoch_train_loss = 0.
                for batch_idx, (batch_data, batch_labels) in enumerate(self.train_loader):
                    self.lr = self._adjust_learning_rate(self.optimizer, self.epochs_run, batch_idx, )
                    batch_data = batch_data.to(self.device)
                    batch_labels = batch_labels.to(self.device)
                    batch_data_logits = self.model(batch_data)
                    self.optimizer.zero_grad()
                    loss = self.loss_criterion(batch_data_logits, batch_labels)
                    loss.backward()
                    total_epoch_train_loss += loss.item() * len(batch_data)
                    if (self.epochs_run % self.cycle_length) + 1 > (self.cycle_length - self.burn_in_epochs
                                                                    - self.num_samples_per_cycle):
                        self.optimizer.step(add_langevin_noise=True)
                    else:
                        self.optimizer.step(add_langevin_noise=False)
                self.epochs_run += 1
                print('Epoch: ', self.epochs_run, ' lr: ', self.lr)
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
                if ((self.epochs_run - 1) % self.cycle_length) >= (self.cycle_length - self.num_samples_per_cycle):
                    sample_collected = True
                    # print('Epoch: ', self.epochs_run, ' lr: ', self.lr)
                    output_model = deepcopy(self.model.cpu())
                    self.model.to(self.device)
                    return output_model


        else:
            raise NotImplementedError

    def sample(self, num_samples=None, val_loader=None, debug_val_loss=False, wandb_debug=False):
        output_list = []
        if num_samples is None:
            num_samples = self.num_samples_per_cycle * self.num_cycles
        if issubclass(self.model.__class__, torch.nn.Module):
            for i in range(num_samples):
                output_list.append(self.sample_iterative(val_loader=val_loader, debug_val_loss=debug_val_loss,
                                                         wandb_debug=wandb_debug))
            return output_list
        else:
            raise NotImplementedError

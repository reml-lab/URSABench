from copy import deepcopy

import torch
import wandb
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR

from URSABench.util import get_loss_criterion, reset_model
from . import optimSGHMC
from .inference_base import _Inference


class SGHMC(_Inference):

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
            hyperparameters = {'lr': 0.001,'prior_std': 10, 'num_samples': 2, 'alpha': 0.1, 'burn_in_epochs':10}

        super(SGHMC, self).__init__(hyperparameters, model, train_loader, device)
        self.lr = hyperparameters['lr']
        self.prior_std = hyperparameters['prior_std']
        self.num_samples = hyperparameters['num_samples']
        self.alpha = hyperparameters['alpha']
        self.burn_in_epochs = hyperparameters['burn_in_epochs']
        self.model_loss = model_loss
        self.model = model
        self.train_loader = train_loader
        self.device = device
        self.dataset_size = len(train_loader.dataset)
        self.optimizer = optimSGHMC(params=self.model.parameters(), lr=self.lr, momentum=1 - self.alpha,
                                    num_training_samples=self.dataset_size, weight_decay=1 / (self.prior_std ** 2))
        self.loss_criterion = get_loss_criterion(loss=model_loss)
        self.burnt_in = False
        self.epochs_run = 0
        self.lr_final = self.lr / 2
        self.optimizer_scheduler = CosineAnnealingLR(optimizer=self.optimizer, T_max=
        (self.burn_in_epochs + self.num_samples),)
#        self.optimizer_scheduler = OneCycleLR(optimizer=self.optimizer, max_lr=self.lr*5,
#                                               steps_per_epoch=len(self.train_loader),
#                                               epochs=self.burn_in_epochs + self.num_samples, cycle_momentum=False)

    def update_hyp(self, hyperparameters):
        self.lr = hyperparameters['lr']
        self.prior_std = hyperparameters['prior_std']
        self.num_samples = hyperparameters['num_samples']
        self.alpha = hyperparameters['alpha']
        self.burn_in_epochs = hyperparameters['burn_in_epochs']
        self.model = reset_model(self.model)
        self.burnt_in = False
        self.epochs_run = 0
        self.optimizer = optimSGHMC(params=self.model.parameters(), lr=self.lr, momentum=1 - self.alpha,
                                    num_training_samples=self.dataset_size, weight_decay=1 / (self.prior_std ** 2))
        self.lr_final = self.lr / 2
        self.optimizer_scheduler = CosineAnnealingLR(optimizer=self.optimizer, T_max=
        self.burn_in_epochs + self.num_samples, eta_min=self.lr_final)

    def sample_iterative(self, val_loader=None, debug_val_loss=False, wandb_debug=False):
        if issubclass(self.model.__class__, torch.nn.Module):
            if self.burnt_in is False:
                epochs = self.burn_in_epochs + 1
                self.burnt_in = True
            else:
                epochs = 1
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
                    if epoch > 0.8 * epochs or self.burnt_in:
                        self.optimizer.step(add_langevin_noise=True)
                    else:
                        self.optimizer.step(add_langevin_noise=False)
                self.optimizer_scheduler.step()
                if debug_val_loss:
                    avg_val_loss = self.compute_val_loss(val_loader)
                    avg_train_loss = total_epoch_train_loss / self.dataset_size
                    metrics = {
                        'train_loss': avg_train_loss,
                        'val_loss': avg_val_loss,
                        'lr': self.optimizer_scheduler.get_lr()
                    }
                    print(metrics)
                    if wandb_debug:
                        wandb.log(metrics)
            output_model = deepcopy(self.model.cpu())
            self.model.to(self.device)
            return output_model
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

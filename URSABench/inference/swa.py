from copy import deepcopy

import torch
import wandb
from torch.optim import SGD

from URSABench.util import get_loss_criterion, reset_model, set_weights, flatten, \
    bn_update, adjust_learning_rate
from .inference_base import _Inference
from .subspaces import Subspace


class SWA(_Inference):
    def __init__(self, hyperparameters, model=None, train_loader=None, model_loss='multi_class_linear_output',
                 device=torch.device('cpu'), **subspace_kwargs):
        super(SWA, self).__init__(hyperparameters, model=None, train_loader=None, device=torch.device('cpu'))
        if hyperparameters == None:
            # Initialise as some default values
            hyperparameters = {'swag_lr': 0.001,'swag_wd': 0.001,'lr_init': 0.001, 'num_samples': 20, 'momentum': 0.1, 'burn_in_epochs':100, 'num_iterates':50}

        self.hyperparameters = hyperparameters
        self.swag_model = deepcopy(model)
        self.model = model
        self.num_parameters = sum(param.numel() for param in self.swag_model.parameters())
        self.weight_mean = torch.zeros(self.num_parameters)
        self.sq_mean = torch.zeros(self.num_parameters)
        self.num_models_collected = torch.zeros(1, dtype=torch.long)
        self.var_clamp = 1e-30
        self.device = device
        self.train_loader = train_loader
        self.loss_criterion = get_loss_criterion(loss=model_loss)
        self.dataset_size = len(train_loader.dataset)
        self.burnt_in = False
        self.epochs_run = 0
        self.burn_in_epochs = self.hyperparameters['burn_in_epochs']
        self.num_iterates = self.hyperparameters['num_iterates']
        self.momentum = self.hyperparameters['momentum']
        self.lr_init = self.hyperparameters['lr_init']
        self.swag_lr = self.hyperparameters['swag_lr']
        self.swag_wd = self.hyperparameters['swag_wd']
        self.optimizer = SGD(params=self.model.parameters(), lr=self.lr_init, momentum=self.momentum,
                             weight_decay=self.swag_wd)
        if 'subspace_type' not in hyperparameters.keys():
            self.subspace_type = 'pca'
        else:
            self.subspace_type = hyperparameters['subspace_type']
        if subspace_kwargs is None:
            subspace_kwargs = dict()
        self.subspace = Subspace.create(self.subspace_type, num_parameters=self.num_parameters,
                                        **subspace_kwargs)
        self.cov_factor = None

    def update_hyp(self, hyperparameters, **subspace_kwargs):
        self.weight_mean = torch.zeros(self.num_parameters)
        self.sq_mean = torch.zeros(self.num_parameters)
        self.num_models_collected = torch.zeros(1, dtype=torch.long)
        self.burnt_in = False
        self.epochs_run = 0
        self.hyperparameters = hyperparameters
        self.burn_in_epochs = self.hyperparameters['burn_in_epochs']
        self.num_iterates = self.hyperparameters['num_iterates']
        self.momentum = self.hyperparameters['momentum']
        self.lr_init = self.hyperparameters['lr_init']
        self.swag_lr = self.hyperparameters['swag_lr']
        self.swag_wd = self.hyperparameters['swag_wd']
        self.model = reset_model(self.model)
        self.swag_model = reset_model(self.swag_model)
        self.optimizer = SGD(params=self.model.parameters(), lr=self.lr_init, momentum=self.momentum,
                             weight_decay=self.swag_wd)
        if 'subspace_type' not in hyperparameters.keys():
            self.subspace_type = 'pca'
        else:
            self.subspace_type = hyperparameters['subspace_type']
        if subspace_kwargs is None:
            subspace_kwargs = dict()
        self.subspace = Subspace.create(self.subspace_type, num_parameters=self.num_parameters,
                                        **subspace_kwargs)

    def _collect_model(self):

        w = flatten([param.detach().cpu() for param in self.model.parameters()])
        # first moment
        self.weight_mean.mul_(self.num_models_collected.item() / (self.num_models_collected.item() + 1.0))
        self.weight_mean.add_(w / (self.num_models_collected.item() + 1.0))

        # second moment
        self.sq_mean.mul_(self.num_models_collected.item() / (self.num_models_collected.item() + 1.0))
        self.sq_mean.add_(w ** 2 / (self.num_models_collected.item() + 1.0))
        deviation_vector = w - self.weight_mean
        self.subspace.collect_vector(deviation_vector)

    def _schedule(self, epoch):
        t = epoch / self.burn_in_epochs
        lr_ratio = self.swag_lr / self.lr_init
        if t <= 0.5:
            factor = 1.0
        elif t <= 0.9:
            factor = 1.0 - (1.0 - lr_ratio) * (t - 0.5) / 0.4
        else:
            factor = lr_ratio
        return self.lr_init * factor

    def _set_swa(self):
        set_weights(self.swag_model, self.weight_mean, self.device)

    def _get_mean_and_variance(self):
        variance = torch.clamp(self.sq_mean - self.weight_mean ** 2, self.var_clamp)
        return self.weight_mean, variance

    def fit(self):
        if self.cov_factor is not None:
            return
        self.cov_factor = self.subspace.get_space()

    def get_space(self, export_cov_factor=True):
        mean, variance = self._get_mean_and_variance()
        if not export_cov_factor:
            return mean.clone(), variance.clone()
        else:
            self.fit()
            return mean.clone(), variance.clone(), self.cov_factor.clone()

    def sample_iterative(self, update_bn_swa=True, val_loader=None, debug_val_loss=False, wandb_debug=False):
        if issubclass(self.model.__class__, torch.nn.Module):
            if self.burnt_in is False:
                epochs = self.burn_in_epochs + 1
                self.burnt_in = True
            else:
                epochs = 1
            self.num_models_collected += 1
            for epoch in range(epochs):
                self.model.train()
                lr = self._schedule(self.epochs_run)
                adjust_learning_rate(self.optimizer, lr)
                total_epoch_train_loss = 0.
                for batch_idx, (batch_data, batch_labels) in enumerate(self.train_loader):
                    batch_data = batch_data.to(self.device)
                    batch_labels = batch_labels.to(self.device)
                    batch_data_logits = self.model(batch_data)
                    loss = self.loss_criterion(batch_data_logits, batch_labels)
                    self.optimizer.zero_grad()
                    loss.backward()
                    total_epoch_train_loss += loss.item() * len(batch_data)
                    self.optimizer.step()
                self.epochs_run += 1
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
            self._collect_model()
            if update_bn_swa:
                self._set_swa()
                bn_update(self.train_loader, self.swag_model)
            return self.swag_model
        else:
            raise NotImplementedError

    def sample(self, num_samples=None, val_loader=None, debug_val_loss=False, wandb_debug=False):
        output_list = []
        if num_samples is None:
            num_samples = self.num_iterates
        if issubclass(self.model.__class__, torch.nn.Module):
            for i in range(num_samples):
                if i == num_samples - 1:
                    output_list.append(self.sample_iterative(update_bn_swa=True, val_loader=val_loader,
                                                             debug_val_loss=debug_val_loss, wandb_debug=wandb_debug))
                else:
                    output_list.append(self.sample_iterative(update_bn_swa=False, val_loader=val_loader,
                                                             debug_val_loss=debug_val_loss, wandb_debug=wandb_debug))
            return output_list
        else:
            raise NotImplementedError

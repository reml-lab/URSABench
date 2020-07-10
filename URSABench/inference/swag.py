from copy import deepcopy

import torch
import wandb
from torch.optim import SGD

from URSABench.util import reset_model, bn_update, adjust_learning_rate
from .subspaces import Subspace
from .swa import SWA


class SWAG(SWA):
    def __init__(self, hyperparameters, model=None, train_loader=None, model_loss='multi_class_linear_output',
                 device=torch.device('cpu'), **subspace_kwargs):
        super(SWAG, self).__init__(hyperparameters, model=model, train_loader=train_loader, model_loss=model_loss,
                                   device=device, **subspace_kwargs)
        if hyperparameters == None:
            # Initialise as some default values
            hyperparameters = {'swag_lr': 0.001,'swag_wd': 0.001,'lr_init': 0.001, 'num_samples': 20, 'momentum': 0.1, 'burn_in_epochs':100, 'num_iterates':50}
        self.num_samples = hyperparameters['num_samples']
        self.weight_variance = None

    def update_hyp(self, hyperparameters, **subspace_kwargs):
        self.weight_mean = torch.zeros(self.num_parameters)
        self.weight_variance = None
        self.sq_mean = torch.zeros(self.num_parameters)
        self.num_models_collected = torch.zeros(1, dtype=torch.long)
        self.burnt_in = False
        self.epochs_run = 0
        self.hyperparameters = hyperparameters
        self.burn_in_epochs = self.hyperparameters['burn_in_epochs']
        self.num_iterates = self.hyperparameters['num_iterates']
        self.num_samples = self.hyperparameters['num_samples']
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

    def sample_iterative(self, update_bn=True, val_loader=None, debug_val_loss=False, wandb_debug=False,
                         full_cov=False):
        if issubclass(self.model.__class__, torch.nn.Module):
            if self.burnt_in is False:
                epochs = self.burn_in_epochs + self.num_iterates
                for epoch in range(epochs):
                    self.model.train()
                    total_epoch_train_loss = 0.
                    lr = self._schedule(self.epochs_run)
                    adjust_learning_rate(self.optimizer, lr)
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
                    if epoch >= self.burn_in_epochs:
                        self._collect_model()
                self.burnt_in = True
                _, self.weight_variance = self._get_mean_and_variance()
                # if full_cov is False:
                #     weight_sample = torch.normal(self.weight_mean, torch.sqrt(self.weight_variance))
                # else:
                #     var_sample = self.weight_variance.sqrt() * torch.randn_like(self.weight_variance,
                #                                                                 requires_grad=False)
                #     cov_sample = self.swag_model.subspace.cov_mat_sqrt.t().matmul(
                #         self.swag_model.subspace.cov_mat_sqrt.new_empty(
                #             (self.swag_model.subspace.cov_mat_sqrt.size(0),), requires_grad=False
                #         ).normal_()
                #     )
                #     cov_sample /= (self.swag_model.subspace.max_rank - 1) ** 0.5
                #     rand_sample = var_sample + cov_sample
                #     weight_sample = self.weight_mean + rand_sample
                weight_sample = self.weight_mean
                offset = 0
                for param in self.swag_model.parameters():
                    param.data.copy_(weight_sample[offset:offset + param.numel()].view(param.size()).to(self.device))
                    offset += param.numel()
            else:
                assert (self.burnt_in is True)
                # if full_cov is False:
                #     weight_sample = torch.normal(self.weight_mean, torch.sqrt(self.weight_variance))
                # else:
                #     var_sample = self.weight_variance.sqrt() * torch.randn_like(self.weight_variance,
                #                                                                 requires_grad=False)
                #     cov_sample = self.swag_model.subspace.cov_mat_sqrt.t().matmul(
                #         self.swag_model.subspace.cov_mat_sqrt.new_empty(
                #             (self.swag_model.subspace.cov_mat_sqrt.size(0),), requires_grad=False
                #         ).normal_()
                #     )
                #     cov_sample /= (self.swag_model.subspace.max_rank - 1) ** 0.5
                #     rand_sample = var_sample + cov_sample
                #     weight_sample = self.weight_mean + rand_sample
                weight_sample = self.weight_mean
                offset = 0
                for param in self.swag_model.parameters():
                    param.data.copy_(weight_sample[offset:offset + param.numel()].view(param.size()).to(self.device))
                    offset += param.numel()
            if update_bn:
                bn_update(self.train_loader, self.swag_model)
            output_model = deepcopy(self.swag_model.cpu())
            self.swag_model.to(self.device)
            return output_model
        else:
            raise NotImplementedError

    def sample(self, num_samples=None, val_loader=None, debug_val_loss=False, wandb_debug=False, full_cov=False):
        output_list = []
        if num_samples is None:
            num_samples = self.num_samples
        if issubclass(self.model.__class__, torch.nn.Module):
            for i in range(num_samples):
                if i == num_samples - 1:
                    output_list.append(self.sample_iterative(update_bn=True, val_loader=val_loader,
                                                             debug_val_loss=debug_val_loss, wandb_debug=wandb_debug,
                                                             full_cov=full_cov))
                else:
                    output_list.append(self.sample_iterative(update_bn=True, val_loader=val_loader,
                                                             debug_val_loss=debug_val_loss, wandb_debug=wandb_debug,
                                                             full_cov=full_cov))
            return output_list
        else:
            raise NotImplementedError

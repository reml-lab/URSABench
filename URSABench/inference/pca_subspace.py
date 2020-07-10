from copy import deepcopy

import numpy as np
import torch
import wandb

from URSABench.inference.inference_base import _Inference
from URSABench.inference.projection_model import SubspaceModel
from URSABench.inference.swa import SWA
from URSABench.util import reset_model, log_pdf, cross_entropy, elliptical_slice, bn_update


class PCASubspaceSampler(_Inference):
    def __init__(self, hyperparameters, model=None, train_loader=None, model_loss='multi_class_linear_output',
                 device=torch.device('cpu')):
        super(PCASubspaceSampler, self).__init__(hyperparameters=hyperparameters,
                                                 model=model, train_loader=train_loader, device=device)
        if hyperparameters == None:
            # Initialise as some default values
            self.hyperparameters = {'swag_lr': 0.001, 'swag_wd': 0.001, 'lr_init': 0.001, 'num_samples': 20,
                                    'swag_momentum': 0.1, 'swag_burn_in_epochs': 100, 'num_swag_iterates': 50,
                                    'rank': 20, 'max_rank': 20, 'temperature': 5000, 'prior_std': 2.}
        else:
            self.hyperparameters = hyperparameters

        self.rank = self.hyperparameters['rank']
        self.max_rank = self.hyperparameters['max_rank']
        self.model = model
        # self.sampler = hyperparameters['ess']
        self.train_loader = train_loader
        self.device = device
        self.lr_init = self.hyperparameters['lr_init']
        # import pdb; pdb.set_trace()
        self.swag_lr = self.hyperparameters['swag_lr']
        self.swag_burn_in_epochs = self.hyperparameters['swag_burn_in_epochs']
        self.num_samples = self.hyperparameters['num_samples']
        self.num_swag_iterates = self.hyperparameters['num_swag_iterates']
        self.swag_momentum = self.hyperparameters['swag_momentum']
        self.lr_init = self.hyperparameters['lr_init']
        self.swag_lr = self.hyperparameters['swag_lr']
        self.swag_wd = self.hyperparameters['swag_wd']
        self.prior_std = self.hyperparameters['prior_std']
        self.temperature = self.hyperparameters['temperature']
        self.model_loss_type = model_loss
        self.subspace_constructed = False
        self.current_theta = None
        swag_hyperparam_dict = {
            'burn_in_epochs': self.swag_burn_in_epochs,
            # 'num_samples': self.num_swag_iterates,
            'momentum': self.swag_momentum,
            'lr_init': self.lr_init,
            'swag_lr': self.swag_lr,
            'swag_wd': self.swag_wd,
            'num_iterates': self.num_swag_iterates,
            'subspace_type': 'pca'
        }
        subspace_kwargs = {
            'max_rank': self.max_rank,
            'pca_rank': self.rank
        }
        self.swag_model = SWA(hyperparameters=swag_hyperparam_dict, model=self.model,
                              train_loader=self.train_loader, model_loss=self.model_loss_type,
                              device=self.device, **subspace_kwargs)
        self.weight_mean = None
        self.weight_covariance = None
        self.subspace = None

    def update_hyp(self, hyperparameters):
        self.rank = hyperparameters['rank']
        self.max_rank = hyperparameters['max_rank']
        # self.sampler = hyperparameters['ess']
        self.lr_init = hyperparameters['lr_init']
        self.swag_lr = hyperparameters['swag_lr']
        self.swag_burn_in_epochs = hyperparameters['swag_burn_in_epochs']
        self.num_samples = hyperparameters['num_samples']
        self.num_swag_iterates = hyperparameters['num_swag_iterates']
        self.swag_momentum = hyperparameters['swag_momentum']
        self.lr_init = hyperparameters['lr_init']
        self.swag_lr = hyperparameters['swag_lr']
        self.swag_wd = hyperparameters['swag_wd']
        # import pdb; pdb.set_trace()
        self.prior_std = hyperparameters['prior_std']
        self.temperature = self.hyperparameters['temperature']
        self.subspace_constructed = False
        self.current_theta = None
        swag_hyperparam_dict = {
            'burn_in_epochs': self.swag_burn_in_epochs,
            'num_iterates': self.num_swag_iterates,
            'momentum': self.swag_momentum,
            'lr_init': self.lr_init,
            'swag_lr': self.swag_lr,
            'swag_wd': self.swag_wd,
            'subspace_type': 'pca'
        }
        subspace_kwargs = {
            'max_rank': self.max_rank,
            'pca_rank': self.rank
        }
        self.model = reset_model(self.model)
        self.swag_model.update_hyp(swag_hyperparam_dict, **subspace_kwargs)
        self.subspace_constructed = False
        self.weight_mean = None
        self.weight_covariance = None
        self.subspace = None

    def _oracle(self, theta, subspace):
        return log_pdf(theta, subspace, self.model, self.train_loader, cross_entropy, self.temperature,
                       self.device)

    def sample_iterative(self, update_bn=True, val_loader=None, debug_val_loss=False, wandb_debug=False):
        if self.subspace_constructed is False:
            self.swag_model.sample(val_loader=val_loader, debug_val_loss=debug_val_loss, wandb_debug=wandb_debug)
            self.subspace_constructed = True
        if self.weight_mean is None or self.weight_covariance is None:
            self.weight_mean, _, self.weight_covariance = self.swag_model.get_space()
        if self.subspace is None:
            self.subspace = SubspaceModel(self.weight_mean, self.weight_covariance)
        if self.current_theta is None:
            self.current_theta = torch.zeros(self.rank)
        prior_sample = np.random.normal(loc=0.0, scale=self.prior_std, size=self.rank)
        theta, log_prob = elliptical_slice(initial_theta=self.current_theta.numpy().copy(), prior=prior_sample,
                                           lnpdf=self._oracle, subspace=self.subspace)
        self.current_theta = torch.FloatTensor(theta)
        weight_sample = self.subspace(self.current_theta)
        offset = 0
        for param in self.model.parameters():
            param.data.copy_(weight_sample[offset:offset + param.numel()].view(param.size()).to(self.device))
            offset += param.numel()
        if debug_val_loss:
            avg_val_loss = self.compute_val_loss(val_loader)
            # avg_train_loss = total_epoch_train_loss / self.dataset_size
            metrics = {
                # 'train_loss': avg_train_loss,
                'val_loss': avg_val_loss
            }
            print(metrics)
            if wandb_debug:
                wandb.log(metrics)
        if update_bn:
            bn_update(self.train_loader, self.model)
        output_model = deepcopy(self.model.cpu())
        self.model.to(self.device)
        return output_model

    def sample(self, num_samples=None, val_loader=None, debug_val_loss=False, wandb_debug=False):
        if num_samples is None:
            num_samples = self.num_samples
        output_model_list = []
        for i in range(num_samples):
            if i == num_samples - 1:
                output_model_list.append(self.sample_iterative(update_bn=True, val_loader=val_loader,
                                                               debug_val_loss=debug_val_loss, wandb_debug=wandb_debug))
            else:
                output_model_list.append(self.sample_iterative(update_bn=False, val_loader=val_loader,
                                                               debug_val_loss=debug_val_loss, wandb_debug=wandb_debug))

        return output_model_list

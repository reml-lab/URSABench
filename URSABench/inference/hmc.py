import sys

import torch

from URSABench.util import reset_model, convert_sample_to_net
from .inference_base import _Inference

if 'hamiltorch' not in sys.modules:
    print('You have not imported the hamiltorch module,\nrun: pip install git+https://github.com/AdamCobb/hamiltorch')
import hamiltorch


# def make_model(sample, model):
#     fmodel = hamiltorch.util.make_functional(model)
#     params_unflattened = hamiltorch.util.unflatten(model, sample)
#     return lambda x : fmodel(x,params=params_unflattened)


# TODO: Refactor docstrings for class below.
class HMC(_Inference):
    """ Basic class for HMC using hamiltorch """

    def __init__(self, hyperparameters, model=None, train_loader=None, model_loss='multi_class_linear_output',
                 device=torch.device('cpu')):
        super(HMC, self).__init__(hyperparameters, model, train_loader, device)
        """
        Inputs:
            hyperparameters: ['step_size', 'num_samples', 'L', 'prior_precision']
            model_loss: Specific to output of model, default linear output (classification)
        """
        if hyperparameters == None:
            # Initialise as some default values
            hyperparameters = {'step_size': 0.001,'num_samples': 10, 'L': 1, 'tau': 0.1, 'burn': -1, 'mass': 1.0}

        self.step_size = hyperparameters['step_size']
        self.num_samples = hyperparameters['num_samples']
        self.L = hyperparameters['L']
        self.tau = hyperparameters['tau']
        self.burn = hyperparameters['burn']
        self.mass = hyperparameters['mass']

        self.model_loss = model_loss

        x_train = [];
        y_train = []
        for batch_idx, (data, target) in enumerate(train_loader):
            x_train.append(data.clone().to(self.device))
            y_train.append(target.clone().to(self.device))
        self.x = torch.cat(x_train)
        self.y = torch.cat(y_train)

    def update_hyp(self, hyperparameters):
        # TODO: Check the hyperparameters ar the right type
        self.step_size = hyperparameters['step_size']
        self.num_samples = hyperparameters['num_samples']
        self.L = hyperparameters['L']
        self.tau = hyperparameters['tau']
        self.mass = hyperparameters['mass']
        self.burn = hyperparameters['burn']
        self.model = reset_model(self.model)

    def sample(self, debug = False):
        if issubclass(self.model.__class__, torch.nn.Module):
            tau_list = []
            for w in self.model.parameters():
                tau_list.append(self.tau)
            tau_list = torch.tensor(tau_list).to(self.device)
            tau_out = 1.  # For Regression make this a hyperparameter
            params_init = hamiltorch.util.flatten(self.model).to(self.device).clone()
            inv_mass = (torch.ones(params_init.shape) / self.mass).to(self.device)
            samples = hamiltorch.sample_model(self.model, self.x, self.y, params_init=params_init,
                                              model_loss=self.model_loss, num_samples=self.num_samples,
                                              burn=-1, inv_mass=inv_mass, step_size=self.step_size,
                                              num_steps_per_sample=self.L, tau_out=tau_out, tau_list=tau_list,
                                              debug=debug)
            model_list = []
            # Do not return initial sample
            if len(samples) != self.L * self.num_samples + 1:
                print('Warning, thinning of sampling not aligned as reject occured in first sample.')
            for sample in samples[self.burn*self.L::self.L]:
                model_list.append(convert_sample_to_net(sample, self.model))
        else:
            raise NotImplementedError

        return model_list#torch.stack(samples)

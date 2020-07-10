import torch

from URSABench.util import reset_model
from . import optimSGHMC
from .csghmc import cSGHMC


# TODO: Add docstrings for classes below.
class cSGLD(cSGHMC):
    def __init__(self, hyperparameters, model=None, train_loader=None, model_loss='multi_class_linear_output',
                 device=torch.device('cpu')):
        '''
        :param hyperparameters: Hyperparameters include {'lr', 'prior_std', 'num_samples'}
        :param model: Pytorch model to run SGLD on.
        :param train_loader: DataLoader for train data
        :param model_loss: Loss function to use for the model. (e.g.: 'multi_class_linear_output')
        '''
        if hyperparameters == None:
            # Initialise as some default values
            hyperparameters = {'lr_0': 0.001000, 'prior_std': 10.1000, 'num_samples_per_cycle': 5, 'cycle_length': 20, 'burn_in_epochs': 5, 'num_cycles': 10, 'alpha': 1.,}
        hyperparameters['alpha'] = 1.
        super(cSGLD, self).__init__(hyperparameters, model, train_loader, model_loss, device)

    def update_hyp(self, hyperparameters):
        self.lr_0 = hyperparameters['lr_0']
        self.prior_std = hyperparameters['prior_std']
        self.num_samples_per_cycle = hyperparameters['num_samples_per_cycle']
        self.cycle_length = hyperparameters['cycle_length']
        self.alpha = 1.
        self.burn_in_epochs = hyperparameters['burn_in_epochs']
        self.num_cycles = hyperparameters['num_cycles']
        self.model = reset_model(self.model)
        self.optimizer = optimSGHMC(params=self.model.parameters(), lr=self.lr_0, momentum=1 - self.alpha,
                                    num_training_samples=self.dataset_size, weight_decay=1 / (self.prior_std ** 2))
        self.burnt_in = False
        self.epochs_run = 0

import torch

from URSABench.util import reset_model
from . import optimSGHMC
from .sghmc import SGHMC


class SGLD(SGHMC):
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

        hyperparameters['alpha'] = 1.
        super(SGLD, self).__init__(hyperparameters, model, train_loader, model_loss, device)

    def update_hyp(self, hyperparameters):
        self.lr = hyperparameters['lr']
        self.prior_std = hyperparameters['prior_std']
        self.num_samples = hyperparameters['num_samples']
        self.alpha = 1.
        self.burn_in_epochs = hyperparameters['burn_in_epochs']
        self.model = reset_model(self.model)
        self.optimizer = optimSGHMC(params=self.model.parameters(), lr=self.lr, momentum=1 - self.alpha,
                                    num_training_samples=self.dataset_size, weight_decay=1 / (self.prior_std ** 2))
        self.burnt_in = False
        self.epochs_run = 0

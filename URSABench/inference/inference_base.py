import sys

import torch

from URSABench.util import get_loss_criterion

if 'hamiltorch' not in sys.modules:
    print('You have not imported the hamiltorch module,\nrun: pip install git+https://github.com/AdamCobb/hamiltorch')


# TODO: Add docstrings for classes below.
class _Inference:
    """ Base class of inference wrapper """

    def __init__(self, hyperparameters, model=None, train_loader=None, device=torch.device('cpu'),
                 model_loss='multi_class_linear_output'):
        """
        Inputs:
            model: torch.nn.model (TODO Check this is flexible to other models)
            hyperparameters: list of hyperparameters in order expected by inference engine e.g. [[0.0], [2., 4.]]
            train_loader: torch.utils.data.DataLoader
            device: default 'cpu'
        """

        self.model = model
        self.hyperparameters = hyperparameters
        self.train_loader = train_loader
        self.device = device
        self.loss_criterion = get_loss_criterion(loss=model_loss)

    def update_hyp(self, hyperparameters):
        """ Update hyperparameters """
        raise NotImplementedError

    def sample_iterative(self):
        """ Sample in an online manner (return a single sample per call) """
        raise NotImplementedError

    def sample(self):
        """
        Sample multiple samples
            Output: Torch Tensor shape (No Samples, No Parameters)
        """
        raise NotImplementedError

    def compute_val_loss(self, val_loader=None):
        with torch.no_grad():
            num_val_samples = 0
            total_loss = 0.
            self.model.eval()
            for batch_idx, (batch_data, batch_labels) in enumerate(val_loader):
                batch_data_logits = self.model(batch_data.to(self.device))
                batch_loss = self.loss_criterion(batch_data_logits, batch_labels.to(self.device))
                num_val_samples += len(batch_data)
                total_loss += batch_loss.item() * len(batch_data)
            return total_loss / num_val_samples

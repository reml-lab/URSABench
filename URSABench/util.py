import copy
import itertools
import json
import math
import os.path
import random
import sys
import time
from io import StringIO

import hamiltorch
import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn import CrossEntropyLoss
from torch.utils.data import Dataset


def set_random_seed(seed=None):
    if seed is None:
        seed = int((time.time() * 1e6) % 1e8)
    global _random_seed
    _random_seed = seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


##### Silent function wrapper #####
# https://stackoverflow.com/questions/8447185/to-prevent-a-function-from-printing-in-the-batch-console-in-python

class NullIO(StringIO):
    def write(self, txt):
        pass


def silent(fn):
    """Decorator to silence functions."""

    def silent_fn(*args, **kwargs):
        saved_stdout = sys.stdout
        sys.stdout = NullIO()
        result = fn(*args, **kwargs)
        sys.stdout = saved_stdout
        return result

    return silent_fn


#################################

def list_to_dic(names, hyp_list):
    hyp_dic = {}
    for n, name in enumerate(names):
        hyp_dic[name] = hyp_list[n]
    return hyp_dic


def prior_loss(model, prior_std):
    prior_loss = 0.0
    for var in model.parameters():
        nn = torch.div(var, prior_std)
        prior_loss += torch.sum(nn * nn)
    return 0.5 * prior_loss


def langevin_noise_loss(model, lr, alpha, device):
    noise_loss = 0.0
    noise_std = (2 / lr * alpha) ** 0.5
    for var in model.parameters():
        means = torch.zeros(var.size()).to(device)
        noise_loss += torch.sum(var * Variable(torch.normal(means, std=noise_std).to(device),
                                               requires_grad=False))
    return noise_loss


def get_loss_criterion(loss='multi_class_linear_output', **kwargs):
    """
    :param loss:
    :param kwargs:
    :return:
    """
    if loss == 'multi_class_linear_output':
        return CrossEntropyLoss(**kwargs)
    else:
        raise NotImplementedError


def reset_model(model):
    """
    :param model: Model whose parameters need to be reset.
    :return: Model with the reinitialized parameters.
    """
    if issubclass(model.__class__, torch.nn.Module):
        for child_module_name, child_module in model.named_children():
            try:
                child_module.reset_parameters()
            # Below Attribute error is to account for inplace activations as they don't have a reset_parameters()
            # method
            except AttributeError:
                continue
        return model
    else:
        raise NotImplementedError


def convert_sample_to_net(flat_tensor, model):
    """Convert a flat 1D pytorch tensor to a model.

    :param type flat_tensor: 1D pytorch tensor of length number of params in model `flat_tensor`.
    :param type model: PyTorch torch.nn.module `model`.
    :return: A model with weights set according to flat_tensor.
    :rtype: torch.nn.module

    """
    net = copy.deepcopy(model)
    p_unflat = hamiltorch.util.unflatten(net, flat_tensor)
    for param_cur, param_new in zip(net.parameters(), p_unflat):
        param_cur.data = param_new
    return net


def central_smoothing(proba, gamma=1e-4):
    """
    Central smoothing as shown in Malinin et al., 2020
    :param proba: Tensor containing the class probability outputs.
    :param gamma: Gamma value to use for smoothing
    :return: Output tensor after central smoothing
    """

    return (1 - gamma) * proba + gamma * 1 / (proba.shape[1])


def compute_predictive_entropy(proba):
    """
    Compute predictive entropy of a probability distribution.
    :param proba: Tensor containing the class probability outputs.
    :return: Tensor containing entropy values.
    """

    return -(proba * torch.log(proba)).sum(dim=-1)


def json_open_from_file(parser, arg):
    if not os.path.exists(arg):
        parser.error("The file %s does not exist!" % arg)
    else:
        with open(arg, encoding='utf-8') as data_file:
            domain = json.loads(data_file.read())
        return domain

def make_dic_json_format(dic):
    # change torch tensors into floats
    for key in dic.keys():
        if type(dic[key]) is torch.Tensor:
            dic[key] = float(dic[key])
    return dic


def flatten(lst):
    """
    :param lst: List of tensors
    :return: A single contiguous and combined array of flattened tensors
    """
    tmp = [i.contiguous().view(-1, 1) for i in lst]
    return torch.cat(tmp).view(-1)


def set_weights(model, vector, device=None):
    offset = 0
    for param in model.parameters():
        param.data.copy_(vector[offset:offset + param.numel()].view(param.size()).to(device))
        offset += param.numel()


def adjust_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def _check_bn(module, flag):
    if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
        flag[0] = True


def check_bn(model):
    flag = [False]
    model.apply(lambda module: _check_bn(module, flag))
    return flag[0]


def reset_bn(module):
    if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
        module.running_mean = torch.zeros_like(module.running_mean)
        module.running_var = torch.ones_like(module.running_var)


def _get_momenta(module, momenta):
    if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
        momenta[module] = module.momentum


def _set_momenta(module, momenta):
    if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
        module.momentum = momenta[module]


def bn_update(loader, model, subset=None, **kwargs):
    """
        BatchNorm buffers update (if any).
        Performs 1 epochs to estimate buffers average using train dataset.

        :param loader: train dataset loader for buffers average estimation.
        :param model: model being update
        :return: None
    """
    if not check_bn(model):
        return
    model.train()
    momenta = {}
    model.apply(reset_bn)
    model.apply(lambda module: _get_momenta(module, momenta))
    n = 0
    num_batches = len(loader)

    with torch.no_grad():
        if subset is not None:
            num_batches = int(num_batches * subset)
            loader = itertools.islice(loader, num_batches)

        for input, _ in loader:
            input = input.cuda(non_blocking=True)
            input_var = torch.autograd.Variable(input)
            b = input_var.data.size(0)

            momentum = b / (n + b)
            for module in momenta.keys():
                module.momentum = momentum

            model(input_var, **kwargs)
            n += b

    model.apply(lambda module: _set_momenta(module, momenta))


def unflatten_like(vector, likeTensorList):
    output_list = []
    current_idx = 0
    for tensor in likeTensorList:
        n = tensor.numel()
        output_list.append(vector[:, current_idx:current_idx + n].view(tensor.shape))
        current_idx += n
    return output_list


def log_pdf(theta, subspace, model, loader, criterion, temperature, device):
    w = subspace(torch.FloatTensor(theta))
    offset = 0
    for param in model.parameters():
        param.data.copy_(w[offset:offset + param.numel()].view(param.size()).to(device))
        offset += param.numel()
    model.train()
    with torch.no_grad():
        loss = 0
        for data, target in loader:
            data = data.to(device)
            target = target.to(device)
            batch_loss, _, _ = criterion(model, data, target)
            loss += batch_loss * data.size()[0]
    return -loss.item() / temperature


def cross_entropy(model, input, target):
    # standard cross-entropy loss function

    output = model(input)

    loss = F.cross_entropy(output, target)

    return loss, output, {}


def elliptical_slice(initial_theta, prior, lnpdf,
                     cur_lnpdf=None, angle_range=None, subspace=None, **kwargs):
    """
    NAME:
       elliptical_slice
    PURPOSE:
       Markov chain update for a distribution with a Gaussian "prior" factored out
    INPUT:
       initial_theta - initial vector
       prior - cholesky decomposition of the covariance matrix
               (like what np.linalg.cholesky returns),
               or a sample from the prior
       lnpdf - function evaluating the log of the pdf to be sampled
       kwargs= parameters to pass to the pdf
       cur_lnpdf= value of lnpdf at initial_theta (optional)
       angle_range= Default 0: explore whole ellipse with break point at
                    first rejection. Set in (0,2*pi] to explore a bracket of
                    the specified width centred uniformly at random.
    OUTPUT:
       new_theta, new_lnpdf
    HISTORY:
       Originally written in matlab by Iain Murray (http://homepages.inf.ed.ac.uk/imurray2/pub/10ess/elliptical_slice.m)
       2012-02-24 - Written - Bovy (IAS)
    """
    D = len(initial_theta)
    if cur_lnpdf is None:
        cur_lnpdf = lnpdf(initial_theta, subspace, **kwargs)

    # Set up the ellipse and the slice threshold
    if len(prior.shape) == 1:  # prior = prior sample
        nu = prior
    else:  # prior = cholesky decomp
        if not prior.shape[0] == D or not prior.shape[1] == D:
            raise IOError("Prior must be given by a D-element sample or DxD chol(Sigma)")
        nu = np.dot(prior, np.random.normal(size=D))
    hh = math.log(np.random.uniform()) + cur_lnpdf

    # Set up a bracket of angles and pick a first proposal.
    # "phi = (theta'-theta)" is a change in angle.
    if angle_range is None or angle_range == 0.:
        # Bracket whole ellipse with both edges at first proposed point
        phi = np.random.uniform() * 2. * math.pi
        phi_min = phi - 2. * math.pi
        phi_max = phi
    else:
        # Randomly center bracket on current point
        phi_min = -angle_range * np.random.uniform()
        phi_max = phi_min + angle_range
        phi = np.random.uniform() * (phi_max - phi_min) + phi_min

    # Slice sampling loop
    while True:
        # Compute xx for proposed angle difference and check if it's on the slice
        xx_prop = initial_theta * math.cos(phi) + nu * math.sin(phi)
        cur_lnpdf = lnpdf(xx_prop, subspace, **kwargs)
        if cur_lnpdf > hh:
            # New point is on slice, ** EXIT LOOP **
            break
        # Shrink slice to rejected point
        if phi > 0:
            phi_max = phi
        elif phi < 0:
            phi_min = phi
        else:
            raise RuntimeError('BUG DETECTED: Shrunk to current position and still not acceptable.')
        # Propose new angle difference
        phi = np.random.uniform() * (phi_max - phi_min) + phi_min
    return (xx_prop, cur_lnpdf)

def increase_data_imbalance(label, dataset, remove_frac = 0.9, deterministic = True):
    """Reduce instances of a particular class in data.

    :param int label: Label of class e.g. 0  `label`.
    :param type dataset: torchvision.dataset `dataset`.
    :param float remove_frac: Fraction of the class to remove from data set `remove_frac`.
    :return: labels and data arrays.
    :rtype: torch.tensor

    """
    mask = dataset.targets == label
    ind = torch.where(dataset.targets == label)[0]
    if deterministic:
        r_i = torch.arange(len(ind))
    else:
        r_i = torch.randperm(len(ind))
    N = len(ind)
    ind_keep = int(N - remove_frac * N)
    mask[ind[r_i[:ind_keep]]] = False
    new_labels = dataset.targets[mask.logical_not()]
    new_data = dataset.data[mask.logical_not()]
    return new_labels, new_data


class TransformableTensorDataset(Dataset):
    r"""Dataset wrapping tensors.

    Each sample will be retrieved by indexing tensors along the first dimension.

    Arguments:
        *tensors (Tensor): tensors that have the same size of the first dimension.
    """

    def __init__(self, *tensors, transform=None):
        assert all(tensors[0].size(0) == tensor.size(0) for tensor in tensors)
        assert (len(tensors) == 2)
        self.tensors = tensors
        self.transform = transform

    def __getitem__(self, index):
        if self.transform is None:
            return tuple(tensor[index] for tensor in self.tensors)
        else:
            return (self.transform(self.tensors[0][index]), self.tensors[1][index])

    def __len__(self):
        return self.tensors[0].size(0)


class DatasetFromSubset(Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform

    def __getitem__(self, index):
        x, y = self.subset[index]
        if self.transform:
            x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.subset)

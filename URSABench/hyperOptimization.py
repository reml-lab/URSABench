import itertools
import time

import torch
from botorch.acquisition import UpperConfidenceBound
from botorch.fit import fit_gpytorch_model
# BayesOpt
from botorch.models import SingleTaskGP
from botorch.optim import initializers
from botorch.optim import optimize_acqf
from botorch.utils import standardize
from gpytorch.constraints.constraints import GreaterThan
from gpytorch.likelihoods.gaussian_likelihood import (
    GaussianLikelihood,
)
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.priors.torch_priors import GammaPrior

from . import util

import json


# class Method(Enum):
#     GRID_SEARCH = 1
#     BAYES_OPT   = 2
#     HMC_NUTS    = 3

class _HypOpt:
    """Base class of hyperparameter optimization.

    :param type obj_fun: Function that evaluates performance of model `obj_fun`.
    :param type obj_fun_args: A list of arguments to the obj_fun `obj_fun_args`.
    :param type domain: List of dictionaries defining the hyperparameters, type and domain e.g. [{'name': 'lr, 'type': 'continuous',  'domain': (0.001, 0.1), 'dimensionality': 1}] `domain`.
    :param type inference: Inference object `inference`.
    :param type seed: Random seed `seed`.

    """

    def __init__(self, obj_instance, domain, inference, iterative_mode = False, seed = 123):

        util.set_random_seed(seed=seed)

        self.seed = seed
        self.obj_instance = obj_instance
        self.iterative_mode = iterative_mode
        self.domain = domain
        self.inference = inference
        self.time = []

    def inference_step(self, hyp, verbose):
        if len(self.time) == 0:
            print('Timer Starting')

        self.inference.update_hyp(hyp)
        # Reset objective object
        self.obj_instance.reset()

        t0 = time.perf_counter()
        if self.iterative_mode:
            raise NotImplementedError
        else:
            if verbose == 0:
                # No print statements from inference function
                silent_inference_method = util.silent(self.inference.sample)
                samples = silent_inference()
            else:
                samples = self.inference.sample()
            obj = self.obj_instance.update_statistics(samples, output_performance=True)
        t1 = time.perf_counter()
        self.time.append(t1-t0)

        return obj

    def run(self):
        raise NotImplementedError

class RandomSearch(_HypOpt):
    """docstrRandomSearch"""

    def __init__(self, obj_instance, domain, inference, N_evaluations = 10, iterative_mode = False, seed = 123):
        super(RandomSearch, self).__init__(obj_instance, domain, inference, iterative_mode, seed)
        self.N_evaluations = N_evaluations
        self.hyp_names = []
        self.bounds = []
        self.hyp_names_vary = []
        self.param_space_vary = []
        self.param_space_vary_type = []
        self.constants = []
        self.hyp_names_constant = []
        self.grid_size = []

        for dom in self.domain:
            self.hyp_names.append(dom['name'])
            if dom['type'] == 'continuous' or dom['type'] == 'discrete':
                self.hyp_names_vary.append(dom['name'])
                self.param_space_vary_type.append(dom['type'])
                if dom['option'] == 'linspace':
                    self.param_space_vary.append('linspace')
                    self.bounds.append(dom['domain'])
                elif dom['option'] == 'logspace':
                    self.param_space_vary.append('logspace')
                    log_dom = torch.log(torch.tensor([dom['domain'][0], dom['domain'][1]]))
                    self.bounds.append([log_dom[0],log_dom[1]])
                else:
                    raise NotImplementedError
            elif dom['type'] == 'constant':
                self.hyp_names_constant.append(dom['name'])
                if (type(dom['domain']) is list) or (type(dom['domain']) is tuple):
                    print('Domain must be single value 1 for "{}" if type set to "constant" '.format(dom['name']))
                self.constants.append(dom['domain'])
            else:
                print('Only implemented for domain type continuous')
                raise NotImplementedError

        self.bounds = torch.tensor(self.bounds).t()

    def convert_to_param_space(self, x):
        scaled_x = x.tolist()
        for i, option in enumerate(self.param_space_vary):
            if option == 'linspace':
                scaled_x[i] = x[i]
            elif option == 'logspace':
                scaled_x[i] = x[i].exp()
            else :
                raise NotImplementedError
            if self.param_space_vary_type[i] == 'discrete':
                scaled_x[i] = int(x[i])
            if type == 'constant':
                scaled_x[i] = self.domain[i]['domain']

        return scaled_x

    def run(self, verbose=0, return_all=0):
        util.set_random_seed(self.seed)
        constant_dic = util.list_to_dic(self.hyp_names_constant, self.constants)

        train_Y = torch.zeros(self.N_evaluations,1)
        train_X = torch.rand(self.N_evaluations, len(self.hyp_names_vary))
        hyp_list = []
        obj_list = []
        for n in range(self.N_evaluations):
            for d, name in enumerate(self.hyp_names_vary):
                train_X[n,d] = (self.bounds[1,d] - self.bounds[0,d])*train_X[n,d] + self.bounds[0,d]
            # Scale input domain to parameter domain
            re_scaled_new_x = self.convert_to_param_space(train_X[n])
            # List to dictionary
            continuous_dic = util.list_to_dic(self.hyp_names_vary, re_scaled_new_x)
            hyp = {**continuous_dic, **constant_dic}

            obj = self.inference_step(hyp, verbose)
            # Add to train_Y
            train_Y[n] = torch.tensor([obj]).unsqueeze(-1)
            if verbose == 1:
                print('Iteration {},\nhypers: {}, obj: {}'.format(n,hyp,train_Y[n]))
            if return_all:
                hyp_list.append(hyp)
                obj_list.append(obj)

        i = torch.argmax(train_Y)
        max_obj = train_Y[i]
        re_scaled_best_x = self.convert_to_param_space(train_X[i])
        continuous_dic = util.list_to_dic(self.hyp_names_vary, re_scaled_best_x)
        best_hyp = {**continuous_dic, **constant_dic}

        if return_all:
            return best_hyp, max_obj, hyp_list, obj_list
        else:
            return best_hyp, max_obj

    def run_parallel(self, dataset, data_path, model, validation, inference_method, task, verbose = 1):
        util.set_random_seed(self.seed)
        constant_dic = util.list_to_dic(self.hyp_names_constant, self.constants)
        train_X = torch.rand(self.N_evaluations, len(self.hyp_names_vary))
        command_list = []
        for n in range(self.N_evaluations):
            for d, name in enumerate(self.hyp_names_vary):
                train_X[n, d] = (self.bounds[1, d] - self.bounds[0, d]) * train_X[n, d] + self.bounds[0, d]
            # Scale input domain to parameter domain
            re_scaled_new_x = self.convert_to_param_space(train_X[n])
            # List to dictionary
            continuous_dic = util.list_to_dic(self.hyp_names_vary, re_scaled_new_x)
            hyp = {**continuous_dic, **constant_dic}

            print('Make sure you are in the URSABench top folder')
            args = ['python', './URSABench/experiment.py']
            args.extend(['--hyperparams',
                         '\"' + str(util.make_dic_json_format(hyp)).replace("'", "\\\"") + '\"'])  # JSON format
            args.extend(['--dataset', dataset])
            args.extend(['--data_path', data_path])
            args.extend(['--model', model])
            args.extend(['--validation', str(validation)])
            args.extend(['--inference_method', inference_method])
            args.extend(['--task', task])
            args.extend(['--seed', str(self.seed)])
            args.extend(['--use_val'])
            # command = ' '.join(args)
            command_list.append(args)
        if verbose == 1:
            print('Sending off {} jobs.'.format(len(command_list)))
        return command_list



#TODO: Write a wrapper to cut off certain runs after a certain time has passed

class GridSearch(_HypOpt):
    """Grid search over hyperparameters.

    :param type obj_fun: Function that evaluates performance of model `obj_fun`.
    :param type obj_fun_args: A list of arguments to the obj_fun `obj_fun_args`.
    :param type domain: List of dictionaries defining the hyperparameters, type and domain e.g. [{'name': 'lr, 'type': 'continuous',  'domain': (0.001, 0.1), 'dimensionality': 1}] `domain`.
    :param type inference: Inference object `inference`.
    :param type model: torch.nn.model (TODO Check this is flexible to other models) `model`.
    :param type grid_size: List of number of iterations to do per for loop e.g. [10,10,10] for 3 hypers `grid_size`.
    :param type seed: Random seed `seed`.
    :attr type grid_steps: Corresponds to locations of evalutaions taken in each hyperparameter domain that builds the grid `grid_steps`.
    :attr type hyp_names: List of hyperparameter names `hyp_names`.
    :attr domain:
    :attr grid_size:

    """

    def __init__(self, obj_instance, domain, inference, grid_size = None, iterative_mode = False, seed = 123):
        super(GridSearch, self).__init__(obj_instance, domain, inference, iterative_mode, seed)

        if grid_size is None:
            raise RuntimeError('grid_size not set')
        if len(grid_size) != len(self.domain):
            raise RuntimeError('length of grid_size ({}) not set to same length as domain ({})'.format(len(grid_size),len(self.domain)))

        self.grid_size = grid_size
        self.grid_steps = []
        self.hyp_names = []
        self.param_types = []

        for dom, it in zip(self.domain, self.grid_size):
            self.hyp_names.append(dom['name'])
            self.param_types.append(dom['type'])
            if dom['type'] == 'continuous' or dom['type'] == 'discrete':
                if dom['option'] == 'linspace':
                    self.grid_steps.append(torch.linspace(dom['domain'][0], dom['domain'][1], it))
                elif dom['option'] == 'logspace':
                    log_dom = torch.log(torch.tensor([dom['domain'][0], dom['domain'][1]]))
                    self.grid_steps.append(torch.exp(torch.linspace(log_dom[0], log_dom[1], it)))
                else:
                    raise NotImplementedError
            elif dom['type'] == 'constant':
                if (type(dom['domain']) is list) or (type(dom['domain']) is tuple):
                    print('Domain must be single value 1 for "{}" if type set to "constant" '.format(dom['name']))
                self.grid_steps.append(torch.tensor([dom['domain']]))
            else:
                print('Only implemented for domain type continuous')
                raise NotImplementedError

    def convert_to_param_type(self, x):
        scaled_x = list(x)
        for i, type in enumerate(self.param_types):
            if type == 'discrete':
                scaled_x[i] = int(x[i])
            if type == 'constant':
                scaled_x[i] = self.domain[i]['domain']

        return scaled_x

    def run(self, verbose=0, return_all=0):
        """Runs grid search.

        :param type verbose: Whether to print intermediate evaluations. Options: 0: No intermediate print statements; 1: Print all evaluation results. `verbose`.
        :param type return_all: Option to return list of all evaluations `return_all`.
        :return: (dictionary of best hyperparameters, Corresponding minimum objective).
        :rtype: returns a dictionary and pytorch tensor. Optional return of list of all evaluations (locations, values)

        """

        i = 0
        max_obj =  - float("Inf")
        hyp_list = []
        obj_list = []
        for hyp in itertools.product(*self.grid_steps):
            hyp_correct_type = self.convert_to_param_type(hyp)
            hyp = util.list_to_dic(self.hyp_names, hyp_correct_type)

            obj = self.inference_step(hyp, verbose)

            if obj > max_obj:
                max_obj = obj
                best_hyp = hyp # Might need to deep copy
            if verbose == 1:
                print('Iteration {}, hypers: {}, obj: {}'.format(i,hyp,obj))
            if return_all:
                hyp_list.append(hyp)
                obj_list.append(obj)
            i+=1
        if return_all:
            return best_hyp, max_obj, hyp_list, obj_list
        else:
            return best_hyp, max_obj

    def run_parallel(self, dataset, data_path, model, validation, inference_method, task, verbose=1):
        command_list = []
        n = 0
        for hyp in itertools.product(*self.grid_steps):
            hyp_correct_type = self.convert_to_param_type(hyp)
            hyp = util.list_to_dic(self.hyp_names, hyp_correct_type)
            print('Make sure you are in the URSABench top folder')
            args = ['python', './URSABench/experiment.py']
            args.extend(['--hyperparams', '\"' +
                         str(util.make_dic_json_format(hyp)).replace("'", "\\\"") + '\"'])  # JSON format
            args.extend(['--dataset', dataset])
            args.extend(['--data_path', data_path])
            args.extend(['--model', model])
            args.extend(['--validation', str(validation)])
            args.extend(['--inference_method', inference_method])
            args.extend(['--task', task])
            args.extend(['--seed', str(self.seed)])
            args.extend(['--use_val'])
            # command = ' '.join(args)
            command_list.append(args)
            n += 1
        if verbose == 1:
            print('Sending off {} jobs.'.format(len(command_list)))
        return command_list


#TODO: Add docstrings for classes below.

class BayesOpt(_HypOpt):
    # """Grid search over hyperparameters.
    #
    # :param type obj_fun: Function that evaluates performance of model `obj_fun`.
    # :param type obj_fun_args: A list of arguments to the obj_fun `obj_fun_args`.
    # :param type domain: List of dictionaries defining the hyperparameters, type and domain e.g. [{'name': 'lr, 'type': 'continuous',  'domain': (0.001, 0.1), 'dimensionality': 1}] `domain`.
    # :param type inference: Inference object `inference`.
    # :param type model: torch.nn.model (TODO Check this is flexible to other models) `model`.
    # :param type grid_size: List of number of iterations to do per for loop e.g. [10,10,10] for 3 hypers `grid_size`.
    # :param type seed: Random seed `seed`.
    # :attr type grid_steps: Corresponds to locations of evalutaions taken in each hyperparameter domain that builds the grid `grid_steps`.
    # :attr type hyp_names: List of hyperparameter names `hyp_names`.
    # :attr domain:
    # :attr grid_size:
    #
    # """

    def __init__(self, obj_instance, domain, inference, time_limit = float("Inf"), N_evaluations = 10, init_evaluations = 3, threshold_evaluations = 70, acq_func = None, acq_func_kwargs = None, optim_kwargs = None, iterative_mode = False, seed = 123):
        super(BayesOpt, self).__init__(obj_instance, domain, inference, iterative_mode, seed)

        if acq_func is None:
            # Default:
            self.acq_func = UpperConfidenceBound
            self.acq_func_kwargs = {'beta': 0.1} #High Beta => High exploration; Low Beta => High exploitation.
            self.optim_kwargs = {'q':1, 'num_restarts':20, 'raw_samples':200}
        else:
            self.acq_func = acq_func
            self.acq_func_kwargs = acq_func_kwargs
            self.optim_kwargs = optim_kwargs

        self.N = N_evaluations
        self.init_evaluations = init_evaluations
        self.hyp_names = []
        self.bounds = []
        self.hyp_names_vary = []
        self.param_space_vary = []
        self.param_space_vary_type = []
        self.constants = []
        self.hyp_names_constant = []
        self.grid_size = []
        self.time_limit = time_limit
        self.threshold_evaluations = threshold_evaluations

        for dom in self.domain:
            self.hyp_names.append(dom['name'])
            if dom['type'] == 'continuous' or dom['type'] == 'discrete':
                self.hyp_names_vary.append(dom['name'])
                self.param_space_vary_type.append(dom['type'])
                self.grid_size.append(self.init_evaluations)
                if dom['option'] == 'linspace':
                    self.param_space_vary.append('linspace')
                    self.bounds.append(dom['domain'])
                elif dom['option'] == 'logspace':
                    self.param_space_vary.append('logspace')
                    log_dom = torch.log(torch.tensor([dom['domain'][0], dom['domain'][1]]))
                    self.bounds.append([log_dom[0],log_dom[1]])
                else:
                    raise NotImplementedError
            elif dom['type'] == 'constant':
                self.hyp_names_constant.append(dom['name'])
                self.grid_size.append(1)
                if (type(dom['domain']) is list) or (type(dom['domain']) is tuple):
                    print('Domain must be single value 1 for "{}" if type set to "constant" '.format(dom['name']))
                self.constants.append(dom['domain'])
            else:
                print('Only implemented for domain type continuous')
                raise NotImplementedError

        self.bounds = torch.tensor(self.bounds).t()

    def convert_to_param_space(self, x):
        scaled_x = x.tolist()
        for i, option in enumerate(self.param_space_vary):
            if option == 'linspace':
                scaled_x[i] = x[i]
            elif option == 'logspace':
                scaled_x[i] = x[i].exp()
            else :
                raise NotImplementedError
            if self.param_space_vary_type[i] == 'discrete':
                scaled_x[i] = int(x[i])
            if type == 'constant':
                scaled_x[i] = self.domain[i]['domain']
        return scaled_x

    def convert_to_domain_space(self, x):
        scaled_x = x.clone()
        for i, option in enumerate(self.param_space_vary):
            if option == 'linspace':
                scaled_x[:,i] = x[:,i]
            elif option == 'logspace':
                scaled_x[:,i] = x[:,i].log()
            else :
                raise NotImplementedError
            if self.param_space_vary_type[i] == 'discrete':
                scaled_x[:,i] = torch.tensor(x[:,i],dtype=torch.uint8)
            if type == 'constant':
                scaled_x[:,i] = self.domain[i]['domain']
        return scaled_x

    # def scale_to_0_1_bounds(self, x):
    #     scaled_x = x.clone()
    #
    #     for d in range(x.shape[1]):
    #         scaled_x[:,d] = (x[:,d] - self.bounds[0,d])/(self.bounds[1,d] - self.bounds[0,d])
    #     return scaled_x
    #
    # def scale_to_bounds(self, x):
    #     scaled_x = x.clone()
    #     for d in range(x.shape[1]):
    #         scaled_x[:,d] = (self.bounds[1,d] - self.bounds[0,d])*x[:,d] + self.bounds[0,d]
    #     return scaled_x

    def optimize_acqf_and_get_observation(self, acq, seed):
        """Optimizes the acquisition function, and returns a new candidate."""
        init = initializers.gen_batch_initial_conditions(acq, self.bounds, options={"seed": seed}, **self.optim_kwargs)
        # optimize
        candidate, acq_value = optimize_acqf(
        acq, bounds=self.bounds, batch_initial_conditions=init, **self.optim_kwargs)
        # observe new value
        new_x = candidate.detach() #self.scale_to_bounds(candidate.detach())
        return new_x

    def initialize_model(self, train_X, train_Y, state_dict=None):
        """Initialise model for BO."""
        # From: https://github.com/pytorch/botorch/issues/179
        noise_prior = GammaPrior(1.1, 0.05)
        noise_prior_mode = (noise_prior.concentration - 1) / noise_prior.rate
        MIN_INFERRED_NOISE_LEVEL = 1e-3
        likelihood = GaussianLikelihood(
            noise_prior=noise_prior,
            noise_constraint=GreaterThan(
                MIN_INFERRED_NOISE_LEVEL,
                transform=None,
                initial_value=noise_prior_mode,
            ),
        )

        # train_x = self.scale_to_0_1_bounds(train_X)
        train_Y = standardize(train_Y)
        gp = SingleTaskGP(train_X, train_Y, likelihood=likelihood)
        mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
        # load state dict if it is passed
        if state_dict is not None:
            gp.load_state_dict(state_dict)
        return mll, gp

    def run(self, verbose = 0, return_all=0, initialisation = 'GridSearch', save_path = None):

        util.set_random_seed(self.seed)

        constant_dic = util.list_to_dic(self.hyp_names_constant, self.constants)

        # Init with grid search
        if initialisation == 'GridSearch':
            if verbose == 1:
                print('Grid Search Initialisation\n')
            gs = GridSearch(self.obj_instance, self.domain, self.inference, self.grid_size, self.iterative_mode, self.seed)
            best_hyp, max_obj, hyp_list, obj_list = gs.run(verbose = verbose, return_all = True)
            # Reshape hyp_list and obj_list into the right shapes for GP
            train_Y = torch.tensor(obj_list).unsqueeze(-1)
            train_X = torch.zeros(len(obj_list),len(self.hyp_names_vary))
            for n, hyp in enumerate(hyp_list):
                for d, name in enumerate(self.hyp_names_vary):
                    train_X[n,d] = hyp[name]
            # ToDO: allow for random init as aswell (sample in domain)
        elif initialisation == 'RandomSearch':
            # Random sample to initialise model:
            if verbose == 1:
                print('Random Search Initialisation\n')
            gs = RandomSearch(self.obj_instance, self.domain, self.inference, self.init_evaluations, self.iterative_mode, self.seed)
            best_hyp, max_obj, hyp_list, obj_list = gs.run(verbose = verbose, return_all = True)
            train_Y = torch.tensor(obj_list).unsqueeze(-1)
            train_X = torch.zeros(len(obj_list),len(self.hyp_names_vary))
            for n, hyp in enumerate(hyp_list):
                for d, name in enumerate(self.hyp_names_vary):
                    train_X[n,d] = hyp[name]

        train_X = self.convert_to_domain_space(train_X)

        if verbose == 1:
            print('\nBayesOpt\n')

        # # Reshape hyp_list and obj_list into the right shapes for GP
        # train_Y = torch.tensor(obj_list).unsqueeze(-1)
        # train_X = torch.zeros(len(obj_list),len(self.hyp_names_vary))
        # for n, hyp in enumerate(hyp_list):
        #     for d, name in enumerate(self.hyp_names_vary):
        #         train_X[n,d] = hyp[name]

        mll, gp = self.initialize_model(train_X, train_Y)
        # fit the model
        fit_gpytorch_model(mll)

        best_Y = torch.zeros(self.N)
        best_X = torch.zeros(self.N,len(self.hyp_names_vary))

        # hyp_list = []
        max_obj =  - float("Inf")
        threshold_count = 0
        for iteration in range(self.N):
            # Set acquisition function
            acq_func = self.acq_func(gp, **self.acq_func_kwargs)
            # Optimise acquistion
            new_x = self.optimize_acqf_and_get_observation(acq_func, seed=iteration+self.seed)
            # Scale input domain to parameter domain
            re_scaled_new_x = self.convert_to_param_space(new_x.squeeze(0))
            # List to dictionary
            continuous_dic = util.list_to_dic(self.hyp_names_vary, re_scaled_new_x)
            hyp = {**continuous_dic, **constant_dic}

            obj = self.inference_step(hyp, verbose)

            # Add new proposal
            train_X = torch.cat([train_X, new_x])
            train_Y = torch.cat([train_Y, torch.tensor([obj]).unsqueeze(-1)])

            # Collect the best so far:
            i = torch.argmax(train_Y)
            best_Y[iteration] = train_Y[i]
            best_X[iteration] = train_X[i]

            # Update Model (Include state dic for faster fitting)
            mll, gp = self.initialize_model(train_X, train_Y, gp.state_dict())

            hyp_list.append(hyp)

            if verbose == 1:
                print('Iteration {},\nhypers: {}, obj: {}'.format(iteration,hyp,obj))

            if sum(self.time) > self.time_limit:
                print('Time Limit Reached after ',i, ' iterations.' )
                break
            if obj > max_obj:
                max_obj = obj
                if save_path is not None:
                    best_hyp = util.make_dic_json_format(hyp)
                    with open(save_path + '.json', 'w') as fout:
                        json.dump(best_hyp, fout)

            if float(best_Y[iteration - 1]) == max_obj and iteration > 1:
                # if previous best is same as current increment counter
                threshold_count +=1
            else:
                # Reset count
                threshold_count = 0
            if threshold_count > self.threshold_evaluations:
                print('Ending BO as no improvement in {} evaluations'.format(self.threshold_evaluations))
                break


        i = torch.argmax(train_Y)
        best_hyp = hyp_list[i]

        if return_all:
            return best_hyp, max_obj, hyp_list, best_Y[:iteration]
        else:
            return best_hyp, max_obj

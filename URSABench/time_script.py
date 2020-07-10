import argparse
import csv
import json
import warnings
import time

import torch

from URSABench import models, inference, tasks, datasets, util

warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser()

parser.add_argument('--dataset', type=str, default='CIFAR10', help='dataset name (default: CIFAR10)')
parser.add_argument('--data_path', type=str, default=None, required=True, metavar='PATH',
                    help='path to datasets location (default: None)')
parser.add_argument('--num_workers', type=int, default=4, metavar='N', help='number of workers (default: 4)')
parser.add_argument('--model', type=str, default=None, required=True, metavar='MODEL',
                    help='model name (default: None)')
parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
# parser.add_argument('--inference_method', type=str, default='HMC', help='Inference Method (default: HMC)')
# parser.add_argument('--hyperparams', type=str, default=None, help='Hyperparameters in JSON format (default:None)')
parser.add_argument('--hyperparams_path', default=None, help="Path to json file containing hyperparams",
                    type=str)
# parser.add_argument('--task', type=str, default='Prediction', help='Downstream task to evaluate (default: Prediction)')
# parser.add_argument('--split_classes', type=int, default=None)
parser.add_argument('--validation', type=float, default=0.2,
                    help='Proportation of training used as validation (default: Prediction)')
parser.add_argument('--use_val', dest = 'use_val', action='store_true', help='use val dataset instead of test (default: False)')
parser.add_argument('--batch_size', type=int, default=128, metavar='N', help='input batch size (default: 128)')
parser.add_argument('--save_path', type=str, default=None, required=True, metavar='PATH',
                    help='path to file to store results (default: None)')
parser.add_argument('--device_num', type=int, default=0, help='Device number to select (default: 0)')


args = parser.parse_args()
util.set_random_seed(args.seed)
if torch.cuda.is_available():
    args.device = torch.device('cuda')
    torch.cuda.set_device(args.device_num)
else:
    args.device = torch.device('cpu')

# import pdb; pdb.set_trace()
# if args.hyperparams is None:
#     hyperparams = args.hyperparams_path
# else:
#     hyperparams = json.loads(args.hyperparams)
model_cfg = getattr(models, args.model)
loaders, num_classes = datasets.loaders(
    args.dataset,
    args.data_path,
    args.batch_size,
    args.num_workers,
    transform_train=model_cfg.transform_train,
    transform_test=model_cfg.transform_test,
    shuffle_train=True,
    use_validation=False,
    val_size=args.validation
)

# loaders['train'].dataset.data = loaders['train'].dataset.data[:300]
# loaders['train'].dataset.targets = loaders['train'].dataset.targets[:300]

train_loader = loaders['train']
test_loader = loaders['test']
num_classes = int(num_classes)

inference_method_list = ['HMC', 'SGLD', 'SGHMC', 'cSGLD', 'cSGHMC', 'SWAG', 'PCA', 'MCdropout', 'SGD', 'PCASubspaceSampler']

timer_dic = {}
S = 3
T = 10

for inference_method in inference_method_list:
    hyperparams = util.json_open_from_file(parser, args.hyperparams_path +inference_method+'_BO.json')

    print(inference_method)
    print('Time for ' + str(S)+ ' sample.')

    if inference_method == 'HMC':
        hyperparams['burn'] = -1
    #     hyperparams['L'] = 1
    if inference_method == 'SWAG':
        hyperparams['burn_in_epochs'] = 1
    if inference_method == 'PCASubspaceSampler':
        hyperparams['swag_burn_in_epochs']
    if inference_method == 'SGHMC' or inference_method == 'SGLD':
        hyperparams['burn_in_epochs'] = 0
    if inference_method == 'cSGHMC'or inference_method == 'cSGLD':
        hyperparams['burn_in_epochs'] = 0
        hyperparams['num_cycles'] = 1
        hyperparams['num_samples_per_cycle'] = S
    if inference_method == 'MCdropout' or inference_method == 'SGD':
        hyperparams['epochs'] = 0

    hyperparams['num_samples'] = S# 30

    inference_scheme = getattr(inference, inference_method)

    t_tensor = torch.zeros(T)
    for t in range(T):
        print('Trial: ',t)
        model = model_cfg.base(*model_cfg.args, num_classes=num_classes, **model_cfg.kwargs).to(args.device)
        inference_object = inference_scheme(hyperparameters=hyperparams, model=model, train_loader=train_loader,
                                            device=args.device)

        silent_inference_method = util.silent(inference_object.sample)

        t0 = time.perf_counter()
        # inference_object.sample(val_loader=test_loader, debug_val_loss=True)
        model_ensemble = silent_inference_method()
        t1 = time.perf_counter()
        t_tensor[t] = t1 - t0

    timer_dic[inference_method + '_mean'] = t_tensor.mean()
    timer_dic[inference_method + '_std'] = t_tensor.std()

    print('Time: ', t_tensor.mean(), ' +- ', t_tensor.std())

timer_dic = util.make_dic_json_format(timer_dic)

with open(args.save_path + '.json', 'w') as fout:
    json.dump(timer_dic, fout)

# task_method = getattr(tasks, args.task)
# task_data_loader = {'in_distribution_test': test_loader}
# metric_list = 'ALL'
#
# t0 = time.perf_counter()
#
#         silent_inference = util.silent(self.inference.sample)
#         samples = silent_inference()
#
# t1 = time.perf_counter()
# self.time.append(t1-t0)

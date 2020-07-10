import argparse
import csv
import json
import warnings

import torch

from URSABench import models, inference, tasks, datasets, util

warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser()

parser.add_argument('--dataset', type=str, default='CIFAR10', help='dataset name (default: CIFAR10)')
parser.add_argument('--data_path', type=str, default=None, required=True, metavar='PATH',
                    help='path to datasets location (default: None)')
parser.add_argument('--num_workers', type=int, default=4, metavar='N', help='number of workers (default: 4)')
parser.add_argument('--num_trials', type=int, default=1, help='number of repeats of each experiment (default: 1)')
parser.add_argument('--model', type=str, default=None, required=True, metavar='MODEL',
                    help='model name (default: None)')
parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
parser.add_argument('--inference_method', type=str, default='HMC', help='Inference Method (default: HMC)')
parser.add_argument('--hyperparams', type=str, default=None, help='Hyperparameters in JSON format (default:None)')
parser.add_argument('--hyperparams_path', default=None, help="Path to json file containing hyperparams",
                    type=lambda x: util.json_open_from_file(parser, x))
parser.add_argument('--task', type=str, default='Prediction', help='Downstream task to evaluate (default: Prediction)')
parser.add_argument('--split_classes', type=int, default=None)
parser.add_argument('--validation', type=float, default=0.2,
                    help='Proportation of training used as validation (default: Prediction)')
parser.add_argument('--use_val', dest = 'use_val', action='store_true', help='use val dataset instead of test (default: False)')
parser.add_argument('--use_dm_imbalance', dest = 'use_dm_imbalance', action='store_true', help='use imbalance data set for DM (retrain) (default: False)')
parser.add_argument('--batch_size', type=int, default=128, metavar='N', help='input batch size (default: 128)')
parser.add_argument('--save_path', type=str, default=None, required=True, metavar='PATH',
                    help='path to file to store results (default: None)')
parser.add_argument('--device_num', type=int, default=0, help='Device number to select (default: 0)')
parser.add_argument('--pretrained_model_path', type=str, default=None, help='Path of the pretrained model')


args = parser.parse_args()
util.set_random_seed(args.seed)
if torch.cuda.is_available():
    args.device = torch.device('cuda')
    torch.cuda.set_device(args.device_num)
else:
    args.device = torch.device('cpu')

# import pdb; pdb.set_trace()
if args.hyperparams is None:
    hyperparams = args.hyperparams_path
else:
    hyperparams = json.loads(args.hyperparams)
model_cfg = getattr(models, args.model)
loaders, num_classes = datasets.loaders(
    args.dataset,
    args.data_path,
    args.batch_size,
    args.num_workers,
    transform_train=model_cfg.transform_train,
    transform_test=model_cfg.transform_test,
    shuffle_train=True,
    use_validation=args.use_val,
    val_size=args.validation,
    split_classes=args.split_classes
)
train_loader = loaders['train']
test_loader = loaders['test']
# loaders['train'].dataset.data = loaders['train'].dataset.data[:30]
# loaders['train'].dataset.targets = loaders['train'].dataset.targets[:30]

num_classes = int(num_classes)
model = model_cfg.base(*model_cfg.args, num_classes=num_classes, **model_cfg.kwargs).to(args.device)
if args.pretrained_model_path is not None:
    model.load_state_dict(torch.load(args.pretrained_model_path))
inference_method = getattr(inference, args.inference_method)
inference_object = inference_method(hyperparameters=hyperparams, model=model, train_loader=train_loader,
                                    device=args.device)
# model_ensemble = inference_object.sample()
#
# for model_idx, model in enumerate(model_ensemble):
#     torch.save(model.state_dict(), args.save_path + 'sghmc_sample_%d.pt' % model_idx)

task_method = getattr(tasks, args.task)
task_data_loader = {'in_distribution_test': test_loader}
task_data_loader_dm = {'decision_data_test': test_loader} # For if we use no imbalance
metric_list = 'ALL'

# IF PART OF HYPOPT
if args.task == 'Prediction' and args.use_val:
    task_object = task_method(dataloader=task_data_loader, num_classes=num_classes, device=args.device,
                              metric_list=metric_list)
    task_object.update_statistics(models=model_ensemble, output_performance=False, smoothing=True)
    task_performance = task_object.get_performance_metrics()
    hyperparam_values = [hyperparams[key] for key in sorted(hyperparams.keys())]
    print(sorted(hyperparams.keys()))
    task_performance_values = [task_performance[key] for key in sorted(task_performance.keys())]
    print(sorted(task_performance.keys()))
    with open('results.csv', 'a+') as csvFile:
        writer = csv.writer(csvFile, dialect='excel')
        writer.writerow([
            args.dataset,
            args.model,
            args.seed,
            args.inference_method,
            args.task,
            args.batch_size,
            *hyperparam_values,
            *task_performance_values
        ])

# IF PART OF TESTING
S = args.num_trials # Number of trials (Random Seeds)
OOD_loaders_list = []

if not args.use_val:
    if args.dataset == 'MNIST':
        # OOD
        data_name = ['FashionMNIST', 'KMNIST']
        for d_name in data_name:
            loaders, _ = datasets.loaders(
                d_name,
                args.data_path + d_name,
                args.batch_size,
                args.num_workers,
                transform_train=model_cfg.transform_train,
                transform_test=model_cfg.transform_test,
                shuffle_train=True,
                use_validation=False,
                val_size=args.validation,
                split_classes=args.split_classes
            )
            ood_d = {}
            ood_d['data'] = d_name
            ood_d['in_distribution_test'] = task_data_loader['in_distribution_test']
            ood_d['out_distribution_test'] = loaders['test']
            OOD_loaders_list.append(ood_d)

    elif args.dataset == 'CIFAR10' or args.dataset == 'CIFAR100':
        # OOD

        data_name = ['STL10', 'SVHN']
        for d_name in data_name:
            loaders, _ = datasets.loaders(
                d_name,
                args.data_path + d_name,
                args.batch_size,
                args.num_workers,
                transform_train=model_cfg.transform_train,
                transform_test=model_cfg.transform_test,
                shuffle_train=True,
                use_validation=False,
                val_size=args.validation,
                split_classes=args.split_classes
            )
            ood_d = {}
            ood_d['data'] = d_name
            ood_d['in_distribution_test'] = task_data_loader['in_distribution_test']
            ood_d['out_distribution_test'] = loaders['test']
            OOD_loaders_list.append(ood_d)

    elif args.dataset == 'TIN':
        # TODO ADD TASK
        pass
    else:
        raise NotImplementedError

    results_dic = {}
    temp_dic = {}
    cost_list = []
    for s in range(S):
        util.set_random_seed(s)
        print('Prediction: ',s)
        inference_object = inference_method(hyperparameters=hyperparams, model=model, train_loader=train_loader,
                                            device=args.device)
        model_ensemble = inference_object.sample()

        # Prediction
        task_object = task_method(dataloader=task_data_loader, num_classes=num_classes, device=args.device, metric_list=metric_list)
        task_object.update_statistics(models=model_ensemble, output_performance=False, smoothing=True)
        task_performance = task_object.get_performance_metrics()
        if not args.use_dm_imbalance and not args.dataset == 'TIN':
            print('Running DM task on balanced data: ',s)
            dec_object = tasks.Decision(dataloader=task_data_loader_dm, num_classes=num_classes, device=args.device)
            dec_object.update_statistics(models=model_ensemble, output_performance=False, smoothing=True)
            dec_result = dec_object.get_performance_metrics()
            cost_list.append(dec_result['True_Cost'])

        if args.dataset == 'TIN':
            pass
        else:
            print('OOD: ',s)
            for ood_data_loader in OOD_loaders_list:
                ood_object = tasks.OODDetection(data_loader=ood_data_loader, num_classes=num_classes,
                                                device=args.device)
                dic_ood = ood_object.update_statistics(model_ensemble, output_performance=True)

                for i, key in enumerate(dic_ood.keys()):
                    if s == 0:
                        temp_dic[key] = [dic_ood[key]]
                    else:
                        temp_dic[key].append(dic_ood[key])
                    if s == S-1:
                        results_dic[key + '_'+ ood_data_loader['data'] +'_mean'] = torch.mean(torch.tensor(temp_dic[key]).float())
                        results_dic[key + '_'+ ood_data_loader['data'] +'_std'] = torch.std(torch.tensor(temp_dic[key]).float())


        for i, key in enumerate(task_object.required_metric_list):
            if s == 0:
                temp_dic[key] = [task_performance[key]]
            else:
                temp_dic[key].append(task_performance[key])
            if s == S-1:
                results_dic[key + '_mean'] = torch.mean(torch.tensor(temp_dic[key]).float())
                results_dic[key + '_std'] = torch.std(torch.tensor(temp_dic[key]).float())
    if not args.dataset == 'TIN':
        if args.use_dm_imbalance:
            # Decision Making:
            cost_list = []
            for s in range(S):
                print('Decision Making SEED: ',s)
                util.set_random_seed(s)
                loaders, num_classes = datasets.loaders(
                    args.dataset,
                    args.data_path,
                    args.batch_size,
                    args.num_workers,
                    transform_train=model_cfg.transform_train,
                    transform_test=model_cfg.transform_test,
                    shuffle_train=True,
                    use_validation=False,
                    val_size=args.validation,
                    split_classes=args.split_classes,
                    imbalance=True
                )
                train_loader = loaders['train']
                test_loader = loaders['test']
                inference_method = getattr(inference, args.inference_method)
                inference_object = inference_method(hyperparameters=hyperparams, model=model, train_loader=train_loader,
                                                    device=args.device)
                model_ensemble = inference_object.sample()
                task_data_loader = {'decision_data_test': test_loader}
                dec_object = tasks.Decision(dataloader=task_data_loader, num_classes=num_classes, device=args.device)
                dec_object.update_statistics(models=model_ensemble, output_performance=False, smoothing=True)
                dec_result = dec_object.get_performance_metrics()
                cost_list.append(dec_result['True_Cost'])

    results_dic['cost_mean'] = torch.mean(torch.tensor(cost_list))
    results_dic['cost_std'] = torch.std(torch.tensor(cost_list))

    hyperparam_values = [hyperparams[key] for key in sorted(hyperparams.keys())]
    task_performance_values = [results_dic[key] for key in sorted(results_dic.keys())]
    print(sorted(results_dic.keys()))
    with open(args.save_path + 'results.csv', 'a+') as csvFile:
        writer = csv.writer(csvFile, dialect='excel')
        writer.writerow([
            args.dataset,
            args.model,
            args.seed,
            args.inference_method,
            args.task,
            args.batch_size,
            *hyperparam_values,
            *task_performance_values
        ])

    print(results_dic)
    torch.save(results_dic, args.save_path + '_tests.npy')

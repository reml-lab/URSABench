import argparse
import subprocess
import warnings

import torch

from URSABench import models, inference, tasks, datasets, util, hyperOptimization

warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser(description='Run parallel hyperparameter optimisation')


parser.add_argument("--domain", required=True, metavar='PATH',
                    help="Path to json file containing domain of hyperparams", type=lambda x: util.json_open_from_file(parser, x))
parser.add_argument('--hyper_opt', type=str, default='BayesOpt', help='Hyperparameter Optimisation Scheme (default: BayesOpt)')
parser.add_argument('--verbose', type=int, default=1, help='Whether to print each iteration (default: 1)')
parser.add_argument('--dataset', type=str, default='CIFAR10', help='dataset name (default: CIFAR10)')
parser.add_argument('--data_path', type=str, default=None, required=True, metavar='PATH',
                    help='path to datasets location (default: None)')
parser.add_argument('--save_path', type=str, default=None, required=True, metavar='PATH',
                    help='path to file to store results (default: None)')
parser.add_argument('--num_workers', type=int, default=4, metavar='N', help='number of workers (default: 4)')
parser.add_argument('--model', type=str, default=None, required=True, metavar='MODEL',
                    help='model name (default: None)')
parser.add_argument('--batch_size', type=int, default=128, metavar='N', help='input batch size (default: 128)')
parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
parser.add_argument('--inference_method', type=str, default='HMC', help='Inference Method (default: HMC)')
parser.add_argument('--task', type=str, default='Prediction', help='Downstream task to evaluate (default: Prediction)')
parser.add_argument('--validation', type = float, default= 0.2, help='Proportation of training used as validation (default: Prediction)')
parser.add_argument('--split_classes', type=int, default=None)
parser.add_argument('--device_num', type=int, default=0, help='Device number to select (default: 0)')
# BayesOpt Specific
parser.add_argument('--N_evaluations', type=int, default='10', help='Number of evaluations for BayesOpt (default: 10)')
parser.add_argument('--init_evaluations', type=int, default='10', help='Number of randomly drawn evaluations for initialiation of BayesOpt (default: 10)')
parser.add_argument('--time_limit', type=float, default='Inf', help='Time limit in seconds before stopping BayesOpt (default: Inf)')

args = parser.parse_args()
util.set_random_seed(args.seed)
if torch.cuda.is_available():
    args.device = torch.device('cuda')
    torch.cuda.set_device(args.device_num)
else:
    args.device = torch.device('cpu')
model_cfg = getattr(models, args.model)
loaders, num_classes = datasets.loaders(
    args.dataset,
    args.data_path,
    args.batch_size,
    args.num_workers,
    transform_train=model_cfg.transform_train,
    transform_test=model_cfg.transform_test,
    shuffle_train=True,
    use_validation=True,
    val_size=args.validation,
    split_classes=args.split_classes
)
train_loader = loaders['train']
test_loader = loaders['test']
num_classes = int(num_classes)
model = model_cfg.base(*model_cfg.args, num_classes=num_classes, **model_cfg.kwargs).to(args.device)
inference_method = getattr(inference, args.inference_method)
inference_object = inference_method(hyperparameters=None, model=model, train_loader=train_loader,
                                    device=args.device)
task_method = getattr(tasks, args.task)
task_data_loader = {'in_distribution_test': test_loader}
task_object = task_method(dataloader=task_data_loader, num_classes=num_classes, device=args.device, metric_list=['ll'])

hyper_opt_class = getattr(hyperOptimization, args.hyper_opt)

if args.hyper_opt == 'RandomSearch':
    hyper_opt = hyper_opt_class(task_object, args.domain, inference_object, N_evaluations=args.N_evaluations,
                                iterative_mode=False, seed=args.seed)
    command_list = hyper_opt.run_parallel(args.dataset, args.data_path, args.model, args.validation,
                                          args.inference_method, args.task, args.verbose)

# import pdb; pdb.set_trace()
for command in command_list:
    subprocess.run(command)
    print(command)
    # command = ' '.join(args)
    # queue_name = np.random.choice(['1080ti-short', '2080ti-short'], p=[.5, .5])
    # print(queue_name)
    # print(command)
    # sbatch(command, job_name=job_name, stdout=stdout, stderr=stderr, mem='32G', cpus_per_task=1, queue=queue_name, gres='gpu:1', time='0-04:00', exclude='node172')




# best_hyp = util.make_dic_json_format(best_hyp)
#
# with open(args.save_path + '.json', 'w') as fout:
#     json.dump(best_hyp, fout)
#
# results = {'best_hyp':best_hyp, 'best_Y': best_Y, 'time': hyper_opt.time, 'args': args}
#
# torch.save(results, args.save_path + '_args.npy')

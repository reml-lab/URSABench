import torch
import torch.nn.functional as F
import torchvision

from ..util import central_smoothing

from .task_base import _Task

__all__ = ['Decision']


def MNIST_cost(num_classes):
    I =torch.eye(num_classes)
    digits = [3,7]
    C = I.clone()
    C[torch.where(I==0)] = 0.1
    C[digits] = 100.0 # Select more important rows with high cost in error of decision
    C[torch.where(I==1)] = 0
    return C

def CIFAR10_cost(num_classes):
    I =torch.eye(num_classes)
    digits = [0,1,8,9] # Plane, automobile, ship, truck
    C = I.clone()
    C[torch.where(I==0)] = 0.1
    C[digits] = 1.0 # Select more important rows with high cost in error of decision
    C[torch.where(I==1)] = 0
    return C

coarse_label = ['apple', # id 0
'aquarium_fish', 'baby', 'bear','beaver','bed','bee','beetle','bicycle','bottle','bowl','boy','bridge','bus','butterfly','camel', 'can', 'castle', 'caterpillar', 'cattle', 'chair',
'chimpanzee', 'clock', 'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur', 'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster', 'house',
'kangaroo', 'computer_keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion', 'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse', 'mushroom', 'oak_tree',
'orange', 'orchid', 'otter', 'palm_tree', 'pear', 'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine', 'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket',
'rose', 'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake', 'spider', 'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table', 'tank', 'telephone',
'television', 'tiger', 'tractor', 'train', 'trout', 'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman', 'worm']

def CIFAR100_cost(num_classes):
    labels = ['tank', 'rocket', 'pickup_truck']
    digits = []
    for i, l in enumerate(coarse_label):
        for c in labels:
            if c == l:
                digits.append(i)
    I = torch.eye(num_classes)
    C = I.clone()
    C[torch.where(I==0)] = 0.1
    C[digits] = 1.0 # Select more important rows with high cost in error of decision
    C[torch.where(I==1)] = 0
    return C
# print(np.array2string(L.numpy()))

# def decision(y_pred=None, cost_mat=None):
#     """
#     torch function to calculate cost from the cost matrix:
#     Inputs:
#         y_pred: predicted values (S,N,D) (samples)
#         loss_mat: matrix of loss values for selecting outputs (D,D)
#     """
#     S,N,D = y_pred.shape
#     A = torch.matmul(y_pred,cost_mat).mean(0) #Integration over samples
#     D = A.argmin(1)
#     return D

def decision_cost(D, y_true, cost_mat=None):
    """
    torch function to calculate cost from the cost matrix:
    Inputs:
        y_true: true values (N,)
        D: Decisions torch tensor of integers (N,)
        loss_mat: matrix of loss values for selecting outputs (D,D)
    """
    return cost_mat[y_true,D].sum()


class Decision(_Task):
    def __init__(self, dataloader, num_classes, device):
        super(Decision, self).__init__(dataloader, num_classes, device)
        self.data_loader = dataloader['decision_data_test']
        self.num_classes = num_classes
        self.device = device
        self.num_samples_collected = 0
        self.ensemble_proba = torch.zeros(len(self.data_loader.dataset), num_classes)
        self.risk = torch.zeros(len(self.data_loader.dataset), self.num_classes)
        self.targets = list()
        for batch_idx, (batch_data, batch_labels) in enumerate(self.data_loader):
            self.targets.append(batch_labels)
        self.targets = torch.cat(self.targets)

        if self.data_loader.dataset.__class__ is torchvision.datasets.mnist.MNIST:
            self.cost_mat = MNIST_cost(self.num_classes)
        elif self.data_loader.dataset.__class__ is torchvision.datasets.cifar.CIFAR10:
            self.cost_mat = CIFAR10_cost(self.num_classes)
        elif self.data_loader.dataset.__class__ is torchvision.datasets.cifar.CIFAR100:
            self.cost_mat = CIFAR100_cost(self.num_classes)
        else:
            raise NotImplementedError

    def reset(self):
        self.num_samples_collected = 0
        self.ensemble_proba = torch.zeros(len(self.data_loader.dataset), self.num_classes)
        self.risk = torch.zeros(len(self.data_loader.dataset), self.num_classes)

    def update_statistics(self, models, output_performance=True, smoothing = True):
        if isinstance(models, list):
            if all(issubclass(model.__class__, torch.nn.Module) for model in models):
                num_models = len(models)
                self.num_samples_collected += num_models
            else:
                raise NotImplementedError
        else:
            if issubclass(models.__class__, torch.nn.Module):
                self.num_samples_collected += 1
            else:
                raise NotImplementedError

        with torch.no_grad():
            start_idx = 0
            for batch_idx, (batch_data, batch_labels) in enumerate(self.data_loader):
                end_idx = start_idx + len(batch_data)
                batch_data = batch_data.to(self.device)
                if isinstance(models, list):
                    for model_idx, model in enumerate(models):
                        model.to(self.device)
                        model.eval()
                        batch_logits = model(batch_data)
                        proba = central_smoothing(F.log_softmax(batch_logits, dim=-1).exp_().cpu())
                        self.ensemble_proba[start_idx: end_idx] += proba
                        self.risk[start_idx: end_idx] += torch.matmul(proba,self.cost_mat)
                        model.to('cpu')
                else:
                    ## Here models indicates a single model.
                    models.to(self.device)
                    models.eval()
                    batch_logits = models(batch_data)
                    proba = central_smoothing(F.log_softmax(batch_logits, dim=-1).exp_().cpu())
                    self.ensemble_proba[start_idx: end_idx] += proba
                    self.risk[start_idx: end_idx] += torch.matmul(proba,self.cost_mat)
                    models.to('cpu')
                start_idx = end_idx
        if output_performance:
            return self.get_performance_metrics(output_performance, smoothing)

    def get_performance_metrics(self, output_performance=False, smoothing = True):
        output_dict = {}
        D = (self.risk / self.num_samples_collected).argmin(1)
        cost = decision_cost(D,self.targets,self.cost_mat)

        output_dict['True_Cost'] = cost
        output_dict['Decision'] = D
        output_dict['Pred_cost'] = self.risk
        return output_dict

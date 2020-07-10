import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

from .task_base import _Task
from ..util import central_smoothing, compute_predictive_entropy

__all__ = ['OODDetection']
# TODO: Add docstrings.
class OODDetection(_Task):
    def __init__(self, data_loader=None, num_classes=None, device=torch.device('cpu')):
        super(OODDetection, self).__init__(data_loader, num_classes, device)
        self.in_distribution_loader = data_loader['in_distribution_test']
        self.out_distribution_loader = data_loader['out_distribution_test']
        self.num_classes = num_classes
        self.device = device
        self.in_distribution_ensemble_proba = torch.zeros(len(self.in_distribution_loader.dataset), num_classes)
        self.out_distribution_ensemble_proba = torch.zeros(len(self.out_distribution_loader.dataset), num_classes)
        self.in_distribution_data_uncertainty = torch.zeros(len(self.in_distribution_loader.dataset))
        self.out_distribution_data_uncertainty = torch.zeros(len(self.out_distribution_loader.dataset))
        self.in_distribution_total_uncertainty = None
        self.out_distribution_total_uncertainty = None
        self.in_distribution_model_uncertainty = None
        self.out_distribution_model_uncertainty = None
        self.num_samples_collected = 0

    def reset(self):
        self.in_distribution_ensemble_proba = torch.zeros(len(self.in_distribution_loader.dataset), self.num_classes)
        self.out_distribution_ensemble_proba = torch.zeros(len(self.out_distribution_loader.dataset), self.num_classes)
        self.in_distribution_data_uncertainty = torch.zeros(len(self.in_distribution_loader.dataset))
        self.out_distribution_data_uncertainty = torch.zeros(len(self.out_distribution_loader.dataset))
        self.in_distribution_total_uncertainty = None
        self.out_distribution_total_uncertainty = None
        self.in_distribution_model_uncertainty = None
        self.out_distribution_model_uncertainty = None
        self.num_samples_collected = 0

    def update_statistics(self, models, output_performance=True):
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
            for batch_idx, (batch_data, batch_labels) in enumerate(self.in_distribution_loader):
                end_idx = start_idx + len(batch_data)
                batch_data = batch_data.to(self.device)
                if isinstance(models, list):
                    for model_idx, model in enumerate(models):
                        model.to(self.device)
                        model.eval()
                        batch_logits = model(batch_data)
                        smoothened_proba = central_smoothing(F.log_softmax(batch_logits, dim=-1).exp_().cpu())
                        self.in_distribution_ensemble_proba[start_idx: end_idx] += smoothened_proba
                        self.in_distribution_data_uncertainty[start_idx: end_idx] += compute_predictive_entropy(
                            smoothened_proba)
                        model.to('cpu')
                else:
                    ## Here models indicates a single model.
                    models.to(self.device)
                    models.eval()
                    batch_logits = models(batch_data)
                    smoothened_proba = central_smoothing(F.log_softmax(batch_logits, dim=-1).exp_().cpu())
                    self.in_distribution_ensemble_proba[start_idx: end_idx] += smoothened_proba
                    self.in_distribution_data_uncertainty[start_idx: end_idx] += compute_predictive_entropy(
                        smoothened_proba)
                    models.to('cpu')
                start_idx = end_idx

            start_idx = 0
            for batch_idx, (batch_data, batch_labels) in enumerate(self.out_distribution_loader):
                end_idx = start_idx + len(batch_data)
                batch_data = batch_data.to(self.device)
                if isinstance(models, list):
                    for model_idx, model in enumerate(models):
                        model.to(self.device)
                        model.eval()
                        batch_logits = model(batch_data)
                        smoothened_proba = central_smoothing(F.log_softmax(batch_logits, dim=-1).exp_().cpu())
                        self.out_distribution_ensemble_proba[start_idx: end_idx] += smoothened_proba
                        self.out_distribution_data_uncertainty[start_idx: end_idx] += compute_predictive_entropy(
                            smoothened_proba)
                        model.to('cpu')
                else:
                    ## Here models indicates a single model.
                    models.to(self.device)
                    models.eval()
                    batch_logits = models(batch_data)
                    smoothened_proba = central_smoothing(F.log_softmax(batch_logits, dim=-1).exp_().cpu())
                    self.out_distribution_ensemble_proba[start_idx: end_idx] += smoothened_proba
                    self.out_distribution_data_uncertainty[start_idx: end_idx] += compute_predictive_entropy(
                        smoothened_proba)
                    models.to('cpu')
                start_idx = end_idx
        if output_performance:
            return self.get_performance_metrics()

    def get_performance_metrics(self):
        self.in_distribution_total_uncertainty = compute_predictive_entropy(
            self.in_distribution_ensemble_proba / self.num_samples_collected
        )
        self.out_distribution_total_uncertainty = compute_predictive_entropy(
            self.out_distribution_ensemble_proba / self.num_samples_collected
        )
        self.in_distribution_model_uncertainty = self.in_distribution_total_uncertainty - \
                                                 self.in_distribution_data_uncertainty / self.num_samples_collected
        self.out_distribution_model_uncertainty = self.out_distribution_total_uncertainty - \
                                                  self.out_distribution_data_uncertainty / self.num_samples_collected
        label_array = np.concatenate([np.ones(len(self.out_distribution_loader.dataset)),
                                      np.zeros(len(self.in_distribution_loader.dataset))])
        total_uncertainty_array = np.concatenate([self.out_distribution_total_uncertainty.numpy(),
                                                  self.in_distribution_total_uncertainty.numpy()])
        model_uncertainty_array = np.concatenate([self.out_distribution_model_uncertainty.numpy(),
                                                  self.in_distribution_model_uncertainty.numpy()])
        total_uncertainty_auroc_score = roc_auc_score(label_array, total_uncertainty_array)
        model_uncertainty_auroc_score = roc_auc_score(label_array, model_uncertainty_array)

        return {
            'total_uncertainty_auroc': total_uncertainty_auroc_score,
            'model_uncertainty_auroc': model_uncertainty_auroc_score
        }

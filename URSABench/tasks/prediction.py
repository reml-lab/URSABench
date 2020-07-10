import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score as auroc, average_precision_score as prauc

from .task_base import _Task
from .. import util

__all__ = ['Prediction']


class Prediction(_Task):
    supported_metric_list = ['error_rate', 'nll', 'll', 'brier_score', 'ece', 'misclass_model_uncertainty_auroc',
                             'misclass_model_uncertainty_aucpr', 'misclass_total_uncertainty_auroc',
                             'misclass_total_uncertainty_aucpr', 'misclass_confidence_auroc',
                             'misclass_confidence_aucpr']

    def __init__(self, dataloader, num_classes, device, metric_list):
        super(Prediction, self).__init__(dataloader, num_classes, device)
        self.data_loader = dataloader['in_distribution_test']
        self.num_classes = num_classes
        self.device = device
        self.num_samples_collected = 0
        self.ensemble_proba = torch.zeros(len(self.data_loader.dataset), num_classes)
        self.expected_data_uncertainty = torch.zeros(len(self.data_loader.dataset))
        self.required_metric_list = self.supported_metric_list if metric_list == 'ALL' else metric_list
        assert all(metric in self.supported_metric_list for metric in self.required_metric_list)
        self.targets = list()
        for batch_idx, (batch_data, batch_labels) in enumerate(self.data_loader):
            self.targets.append(batch_labels)
        self.targets = torch.cat(self.targets)

    def reset(self):
        self.num_samples_collected = 0
        self.ensemble_proba = torch.zeros(len(self.data_loader.dataset), self.num_classes)

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
                        model.eval()#
                        batch_logits = model(batch_data)
                        self.ensemble_proba[start_idx: end_idx] += F.log_softmax(batch_logits, dim=-1).exp_().cpu()
                        self.expected_data_uncertainty[start_idx: end_idx] += \
                            util.compute_predictive_entropy(util.central_smoothing(
                                F.log_softmax(batch_logits, dim=-1).exp_().cpu()))
                        model.to('cpu')
                else:
                    ## Here models indicates a single model.
                    models.to(self.device)
                    models.eval()
                    batch_logits = models(batch_data)
                    self.ensemble_proba[start_idx: end_idx] += F.log_softmax(batch_logits, dim=-1).exp_().cpu()
                    self.expected_data_uncertainty[start_idx: end_idx] += \
                        util.compute_predictive_entropy(util.central_smoothing(
                            F.log_softmax(batch_logits, dim=-1).exp_().cpu()))
                    models.to('cpu')
                start_idx = end_idx
        if output_performance:
            return self.get_performance_metrics(output_performance, smoothing)

    def get_performance_metrics(self, output_performance=False, smoothing = True):
        output_dict = {}
        for metric in self.required_metric_list:
            if metric == 'error_rate':
                accuracy = np.mean(np.argmax(self.ensemble_proba.numpy() / self.num_samples_collected, axis=1) ==
                                   self.targets.numpy())
                output_dict[metric] = 1 - accuracy
            if metric == 'nll' or metric == 'll':
                if smoothing:
                    nll = F.nll_loss(
                        torch.log(util.central_smoothing(self.ensemble_proba / self.num_samples_collected)),
                        self.targets)
                else:
                    nll = F.nll_loss(torch.log(self.ensemble_proba / self.num_samples_collected), self.targets)
                if metric == 'll':
                    output_dict[metric] = - nll.item()
                else:
                    output_dict[metric] = nll.item()
            if metric == 'brier_score':
                output_dict[metric] = _get_brier((self.ensemble_proba / self.num_samples_collected).numpy(),
                                                 self.targets.numpy())
            if metric == 'ece':
                output_dict[metric] = _get_ece((self.ensemble_proba / self.num_samples_collected).numpy(),
                                               self.targets.numpy())
            if metric == 'misclass_model_uncertainty_auroc':
                output_dict[metric] = _get_misclass_auroc(
                    util.central_smoothing(self.ensemble_proba / self.num_samples_collected).numpy(),
                    self.targets.numpy(), criterion='model_uncertainty', topk=1,
                    expected_data_uncertainty_array=(
                            self.expected_data_uncertainty / self.num_samples_collected).numpy())
            if metric == 'misclass_model_uncertainty_aucpr':
                output_dict[metric] = _get_misclass_aucpr(
                    util.central_smoothing(self.ensemble_proba / self.num_samples_collected).numpy(),
                    self.targets.numpy(), criterion='model_uncertainty', topk=1,
                    expected_data_uncertainty_array=(
                            self.expected_data_uncertainty / self.num_samples_collected).numpy())

            if metric == 'misclass_total_uncertainty_auroc':
                output_dict[metric] = _get_misclass_auroc(
                    util.central_smoothing(self.ensemble_proba / self.num_samples_collected).numpy(),
                    self.targets.numpy(), criterion='entropy', topk=1,
                    expected_data_uncertainty_array=(
                            self.expected_data_uncertainty / self.num_samples_collected).numpy())

            if metric == 'misclass_total_uncertainty_aucpr':
                output_dict[metric] = _get_misclass_aucpr(
                    util.central_smoothing(self.ensemble_proba / self.num_samples_collected).numpy(),
                    self.targets.numpy(), criterion='entropy', topk=1,
                    expected_data_uncertainty_array=(
                            self.expected_data_uncertainty / self.num_samples_collected).numpy())

            if metric == 'misclass_confidence_auroc':
                output_dict[metric] = _get_misclass_auroc(
                    util.central_smoothing(self.ensemble_proba / self.num_samples_collected).numpy(),
                    self.targets.numpy(), criterion='confidence', topk=1,
                    expected_data_uncertainty_array=(
                            self.expected_data_uncertainty / self.num_samples_collected).numpy())

            if metric == 'misclass_confidence_aucpr':
                output_dict[metric] = _get_misclass_aucpr(
                    util.central_smoothing(self.ensemble_proba / self.num_samples_collected).numpy(),
                    self.targets.numpy(), criterion='confidence', topk=1,
                    expected_data_uncertainty_array=(
                            self.expected_data_uncertainty / self.num_samples_collected).numpy())

        if output_performance:
            if len(self.required_metric_list) != 1:
                raise RuntimeError('Multiple metrics in metric list not suitable for output_performance = True')
            return float(output_dict[self.required_metric_list[0]])
        else:
            return output_dict


def _get_ece(preds, targets, n_bins=15):
    """
    ECE ported from Asukha et al., 2020.
    :param preds: Prediction probabilities in a Numpy array
    :param targets: Targets in a numpy array
    :param n_bins: Total number of bins to use.
    :return: Expected calibration error.
    """
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    confidences, predictions = np.max(preds, 1), np.argmax(preds, 1)
    accuracies = (predictions == targets)

    ece = 0.0
    avg_confs_in_bins = []
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = np.logical_and(confidences > bin_lower, confidences <= bin_upper)
        prop_in_bin = np.mean(in_bin)
        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(accuracies[in_bin])
            avg_confidence_in_bin = np.mean(confidences[in_bin])
            delta = avg_confidence_in_bin - accuracy_in_bin
            avg_confs_in_bins.append(delta)
            ece += np.abs(delta) * prop_in_bin
        else:
            avg_confs_in_bins.append(None)
    # For reliability diagrams, also need to return these:
    # return ece, bin_lowers, avg_confs_in_bins
    return ece


def _get_brier(preds, targets):
    """
    Function to compute Brier score as ported from Asukha et al., 2020.
    :param preds: Prediction probabilities in a numpy array
    :param targets: Targets in a numpy array
    :return: Brier score.
    """
    one_hot_targets = np.zeros(preds.shape)
    one_hot_targets[np.arange(len(targets)), targets] = 1.0
    return np.mean(np.sum((preds - one_hot_targets) ** 2, axis=1))


def _misclass_tgt(output, target, topk=(1,)):
    """
    Internal method for misclassification detection.
    :param output: Prediction probabilities as a torch.Tensor
    :param target: Targets as a torch.Tensor
    :param topk: Top-k class-probabilities to consider.
    :return:
    """
    output = torch.Tensor(output)
    target = torch.LongTensor(target)
    with torch.no_grad():
        maxk = max(topk)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].float().sum(0)
            res.append(correct_k)

        return res[0].numpy()


def _get_misclass_auroc(preds, targets, criterion, topk=1, expected_data_uncertainty_array=None):
    """
    Get AUROC for Misclassification detection
    :param preds: Prediction probabilities as numpy array
    :param targets: Targets as numpy array
    :param criterion: Criterion to use for scoring on misclassification detection.
    :param topk: Top-kl class probabilities to consider while making predictions.
    :param expected_data_uncertainty_array: Expected data uncertainty as numpy array
    :return: AUROC on misclassification detection
    """
    misclassification_targets = (1 - _misclass_tgt(preds, targets, (topk,))).astype(bool)

    if criterion == 'entropy':
        criterion_values = np.sum(-preds * np.log(preds), axis=1)
    elif criterion == 'confidence':
        criterion_values = -preds.max(axis=1)
    elif criterion == 'model_uncertainty':
        criterion_values = np.sum(-preds * np.log(preds), axis=1) - expected_data_uncertainty_array
    else:
        raise NotImplementedError

    return auroc(misclassification_targets, criterion_values)


def _get_misclass_aucpr(preds, targets, criterion, topk=1, expected_data_uncertainty_array=None):
    """
    Get AUPRC for Misclassification detection
    :param preds: Prediction probabilities as numpy array
    :param targets: Targets as numpy array
    :param criterion: Criterion to use for scoring on misclassification detection.
    :param topk: Top-kl class probabilities to consider while making predictions.
    :param expected_data_uncertainty_array: Expected data uncertainty as numpy array
    :return: AUPRC on misclassification detection
    """
    misclassification_targets = (1 - _misclass_tgt(preds, targets, (topk,))).astype(bool)

    if criterion == 'entropy':
        criterion_values = np.sum(-preds * np.log(preds), axis=1)
    elif criterion == 'confidence':
        criterion_values = -preds.max(axis=1)
    elif criterion == 'model_uncertainty':
        criterion_values = np.sum(-preds * np.log(preds), axis=1) - expected_data_uncertainty_array
    else:
        raise NotImplementedError

    return prauc(misclassification_targets, criterion_values)

import torch
import torch.nn.functional as F

from .task_base import _Task

__all__ = ['PGD_attack']


class PGD_attack(_Task):

    def __init__(self, dataloader, num_classes, device, metric_list, alpha=1e-2, attack_iters=40, l_inf_norm=0.1):
        super(PGD_attack, self).__init__(dataloader, num_classes, device)
        self.alpha = alpha
        self.attack_iters = attack_iters
        self.l_inf_norm = l_inf_norm
        self.data_loader = dataloader['in_distribution_test']
        self.num_classes = num_classes
        self.device = device
        self.num_samples_collected = 0
        self.ensemble_proba = torch.zeros(len(self.data_loader.dataset), num_classes)
        self.targets = list()
        for batch_idx, (batch_data, batch_labels) in enumerate(self.data_loader):
            self.targets.append(batch_labels)
        self.targets = torch.cat(self.targets)

    def reset(self):
        self.num_samples_collected = 0
        self.ensemble_proba = torch.zeros(len(self.data_loader.dataset), self.num_classes)

    def generate_PGD_adversarial_examples(self, models):
        """
        Input : Models
        [Used Class Variables:
            -  Input_example,
            -  variables defining pgd attack
                - step_size(alpha)
                - l_inf_norm_bound(l_inf_norm)
                - number of iterations(attack_iters)
        ]
        Output : adeversarial examples
        """

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

        output_adversarial_examples = list()
        start_idx = 0

        for batch_idx, (batch_data, batch_labels) in enumerate(self.data_loader):

            delta = torch.zeros_like(batch_data, requires_grad=True)
            # delta will be updated in every attack iteration

            for t in range(self.attack_iters):
                end_idx = start_idx + len(batch_data)
                batch_data = batch_data.to(self.device)

                if isinstance(models, list):
                    for model_idx, model in enumerate(models):
                        model.to(self.device)
                        batch_logits = model(batch_data + delta)
                        self.ensemble_proba[start_idx: end_idx] += F.log_softmax(batch_logits, dim=-1).exp_().cpu()
                        model.to('cpu')
                else:
                    # Here models indicates a single model.
                    models.to(self.device)
                    batch_logits = models(batch_data + delta)
                    self.ensemble_proba[start_idx: end_idx] += F.log_softmax(batch_logits, dim=-1).exp_().cpu()
                    models.to('cpu')

                targets_this_batch = self.targets[start_idx: end_idx]
                # Keeping reduction method 'none' to get loss
                # contibution of every data-case individually
                log_likelihood = F.nll_loss(torch.log(self.ensemble_proba[start_idx: end_idx]/self.num_samples_collected), targets_this_batch, reduction='none')
                log_likelihood.backward()
                # Note: Rather than standard gradient descent, (normalized) steepest descent has been used here
                delta.data = (delta + self.alpha*delta.grad.detach().sign()).clamp(-self.l_inf_norm, self.l_inf_norm)
                delta.grad.zero_()

            # Adding value of delta after all the attack iterations
            batch_data = batch_data + delta.detach()
            output_adversarial_examples.append(batch_data)
            start_idx = end_idx

        output_adversarial_examples = torch.cat(output_adversarial_examples)

        return output_adversarial_examples

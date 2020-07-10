import torch


class _Task:
    def __init__(self, data_loader=None, num_classes=None, device=torch.device('cpu')):
        self.data_loader = data_loader
        self.num_classes = num_classes
        self.device = device

    def reset(self):
        raise NotImplementedError

    def update_statistics(self, model, output_performance=False):
        raise NotImplementedError

    def ensemble_update_statistics(self, model_list, output_performance=False):
        raise NotImplementedError

    def get_performance_metrics(self):
        raise NotImplementedError

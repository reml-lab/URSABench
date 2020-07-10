import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import transforms

__all__ = ['MLP200MNIST', 'MLP_dropout', 'MLP400MNIST', 'MLP600MNIST']

class MLP(nn.Module):
    def __init__(self, hidden_size, input_dim, num_classes):
        super(MLP, self).__init__()
        self.input_dim = input_dim
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.fc1 = nn.Linear(input_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = x.view(-1, self.input_dim)
        x = self.fc1(x)
        x = self.fc2(F.relu(x))
        x = self.fc3(F.relu(x))
        return x

class MLP_dropout(nn.Module):
    def __init__(self, hidden_size, input_dim, num_classes, dropout=0.2):
        super(MLP_dropout, self).__init__()
        self.input_dim= input_dim
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.fc1 = nn.Linear(input_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, num_classes)
        self.dropout = dropout

    def forward(self, x):
        x = x.view(-1, self.input_dim)
        x = self.fc1(x)
        x = self.fc2(F.relu(F.dropout(x, p=self.dropout)))
        x = self.fc3(F.relu(F.dropout(x, p=self.dropout)))
        return x

class Base:
    base = MLP
    args = list()
    kwargs = dict()
    transform_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

class Base_dropout(Base):
    base = MLP_dropout

class MLP200MNIST(Base):
    kwargs = {'hidden_size': 200, 'input_dim':784}

class MLP200MNIST_dropout(Base_dropout):
    kwargs = {'hidden_size': 200, 'input_dim':784, 'dropout': 0.2}

class MLP400MNIST(Base):
    kwargs = {'hidden_size': 400, 'input_dim':784}

class MLP600MNIST(Base):
    kwargs = {'hidden_size': 600, 'input_dim':784}

__version__ = '0.0.0.dev1-private'

from .util import set_random_seed
from .hyperOptimization import GridSearch, BayesOpt
from .inference import HMC, SGLD, SGHMC
from .models import *
from .tasks import *
from .datasets import *

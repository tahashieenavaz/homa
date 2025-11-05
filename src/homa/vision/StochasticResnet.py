import torch
from .Resnet import Resnet
from .StochasticClassifier import StochasticClassifier
from .utils import replace_activations


class StochasticResnet(Resnet, StochasticClassifier):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        replace_activations(self.network, torch.nn.ReLU, self._activation_pool)

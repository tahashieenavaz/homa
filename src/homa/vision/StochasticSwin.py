import torch
from .Swin import Swin
from .StochasticClassifier import StochasticClassifier
from .utils import replace_activations


class StochasticSwin(Swin, StochasticClassifier):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        replace_activations(self.network, torch.nn.GELU, self._activation_pool)
        replace_activations(self.network, torch.nn.ReLU, self._activation_pool)

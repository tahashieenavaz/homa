import torch
from .Swin import Swin
from .StochasticClassifier import StochasticClassifier
from ..common.utils import replace_modules


class StochasticSwin(Swin, StochasticClassifier):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        replace_modules(self.network, torch.nn.GELU, self._activation_pool)
        replace_modules(self.network, torch.nn.ReLU, self._activation_pool)

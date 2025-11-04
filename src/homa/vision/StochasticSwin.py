import torch
from .Swin import Swin
from .StochasticClassifier import StochasticClassifier


class StochasticSwin(Swin, StochasticClassifier):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.replace_activations(torch.nn.GELU)

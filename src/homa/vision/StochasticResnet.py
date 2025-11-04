import torch
from .Resnet import Resnet
from .StochasticClassifier import StochasticClassifier


class StochasticResnet(Resnet, StochasticClassifier):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.replace_activations(torch.nn.ReLU)

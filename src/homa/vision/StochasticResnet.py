from .Resnet import Resnet
from .utils import replace_relu
from ..activations import StochasticActivation


class StochasticResnet(Resnet):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        replace_relu(self.network, StochasticActivation)

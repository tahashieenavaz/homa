from .ResnetModule import ResnetModule
from ..utils import replace_relu
from ...activations import StochasticActivation


class StochasticResnetModule(ResnetModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        replace_relu(self, StochasticActivation)

from .Swin import Swin
from .utils import replace_gelu
from ..activations import StochasticActivation


class StochasticSwin(Swin):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        replace_gelu(self.network.encoder, StochasticActivation)

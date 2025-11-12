import torch
from ...AdaptiveActivationFunction import AdaptiveActivationFunction
from ...concerns import ChannelBased


class FReLU(AdaptiveActivationFunction, ChannelBased):
    def __init__(self):
        super().__init__()
        self.b = None

    def forward(self, x: torch.Tensor):
        self.initialize(x, "b")
        b = self.b.view(self.parameter_shape(x))
        return torch.where(x >= 0, x + b, b)

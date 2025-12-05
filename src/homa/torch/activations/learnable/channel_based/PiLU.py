import torch
from ...AdaptiveActivationFunction import AdaptiveActivationFunction
from ...concerns import ChannelBased


class PiLU(AdaptiveActivationFunction, ChannelBased):
    def __init__(self):
        super().__init__()
        self.a = None
        self.b = None
        self.c = None

    def forward(self, x: torch.Tensor):
        self.initialize(x, ["a", "b", "c"], [1, 0.01, 1])
        a = self.a.view(self.parameter_shape(x))
        b = self.b.view(self.parameter_shape(x))
        c = self.c.view(self.parameter_shape(x))
        return torch.where(x >= c, a * x + c * (1 - a), b * x + c * (1 - b))

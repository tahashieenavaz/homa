import torch
from ...AdaptiveActivationFunction import AdaptiveActivationFunction
from ...concerns import ChannelBased


class DualLine(AdaptiveActivationFunction, ChannelBased):
    def __init__(self):
        super().__init__()
        self.a = None
        self.b = None
        self.m = None

    def forward(self, x: torch.Tensor):
        self.initialize(x, ["a", "b", "m"], [1, 0.01, -0.22])
        a = self.a.view(self.parameter_shape(x))
        b = self.b.view(self.parameter_shape(x))
        m = self.m.view(self.parameter_shape(x))
        return torch.where(x >= 0, a * x + m, b * x + m)

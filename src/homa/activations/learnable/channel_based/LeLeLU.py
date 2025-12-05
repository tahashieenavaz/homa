import torch
from ...AdaptiveActivationFunction import AdaptiveActivationFunction
from ...concerns import ChannelBased


class LeLeLU(AdaptiveActivationFunction, ChannelBased):
    def __init__(self):
        super().__init__()
        self.a = None

    def forward(self, x: torch.Tensor):
        self.initialize(x, "a")
        a = self.a.view(self.parameter_shape(x))
        return torch.where(x >= 0, a * x, 0.01 * a * x)

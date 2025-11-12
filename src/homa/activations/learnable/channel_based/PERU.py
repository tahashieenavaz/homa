import torch
from ...AdaptiveActivationFunction import AdaptiveActivationFunction
from ...concerns import ChannelBased


class PERU(AdaptiveActivationFunction, ChannelBased):
    def __init__(self):
        super().__init__()
        self.a = None
        self.b = None

    def forward(self, x: torch.Tensor):
        self.initialize(x, ["a", "b"])
        a = self.a.view(self.parameter_shape(x))
        b = self.b.view(self.parameter_shape(x))
        return torch.where(x >= 0, a * x, a * x * torch.exp(b * x))

import torch
from ..ActivationFunction import ActivationFunction
from .concerns import ChannelBased


class StarReLU(ActivationFunction, ChannelBased):
    def __init__(self):
        super().__init__()
        self.a = None
        self.b = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.initialize(x, ["a", "b"])
        a = self.a.view(self.parameter_shape(x))
        b = self.b.view(self.parameter_shape(x))
        return a * torch.relu(x).pow(2) + b

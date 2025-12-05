import torch
from .ActivationFunction import ActivationFunction


class ReSech(ActivationFunction):
    def __init__(self):
        super().__init__()

    def sech(self, x: torch.Tensor) -> torch.Tensor:
        return torch.tensor(2) / (torch.exp(x) + torch.exp(-x))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.sech(x) * x

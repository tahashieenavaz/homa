import torch
from .ActivationFunction import ActivationFunction


class SaRa(ActivationFunction):
    def __init__(self):
        super().__init__()
        self.alpha = 0.5
        self.beta = 0.7

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        delta = 1 + self.alpha * torch.exp(-self.beta * x)
        return torch.where(x >= 0, x, x / delta)

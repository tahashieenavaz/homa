import torch
from .ActivationFunction import ActivationFunction


class IpLU(ActivationFunction):
    def __init__(self, alpha: float = 1.4):
        super().__init__()
        self.alpha = alpha

    def forward(self, x: torch.Tensor):
        delta = 1 + x.abs().pow(self.alpha)
        return torch.where(x >= 0, x, x / delta)

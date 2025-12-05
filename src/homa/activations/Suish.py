import torch
from .ActivationFunction import ActivationFunction


class Suish(ActivationFunction):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a = x * torch.exp(-torch.abs(x))
        return torch.max(x, a)

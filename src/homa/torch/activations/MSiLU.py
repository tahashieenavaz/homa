import torch
from .ActivationFunction import ActivationFunction


class MSiLU(ActivationFunction):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        alpha = torch.exp(-x.pow(2) - 1)
        delta = alpha / 4
        return x * torch.sigmoid(x) + delta

import torch
from .ActivationFunction import ActivationFunction


class DerivativeSiLU(ActivationFunction):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a = torch.sigmoid(x)
        b = 1 - torch.sigmoid(x)
        c = 1 + x * b
        return a * c

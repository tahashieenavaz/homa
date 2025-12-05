import torch
from .ActivationFunction import ActivationFunction


class PolyLU(ActivationFunction):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor):
        return torch.where(x >= 0, x, 1 / (1 - x) - 1)

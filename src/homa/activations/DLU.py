import torch
from .ActivationFunction import ActivationFunction


class DLU(ActivationFunction):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor):
        return torch.where(x >= 0, x, x / (1 - x))

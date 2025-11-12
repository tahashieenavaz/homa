import torch
from .ActivationFunction import ActivationFunction


class ASiLU(ActivationFunction):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        alpha = 1 / (1 + torch.exp(-x))
        return torch.arctan(x * alpha)

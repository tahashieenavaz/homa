import torch
from .ActivationFunction import ActivationFunction


class MinSin(ActivationFunction):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.min(x, torch.sin(x))

import torch
from .ActivationFunction import ActivationFunction


class ThLU(ActivationFunction):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.where(x >= 0, x, torch.tanh(x / 2))

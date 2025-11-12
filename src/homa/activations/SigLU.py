import torch
from .ActivationFunction import ActivationFunction


class SigLU(ActivationFunction):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        delta = torch.exp(-2 * x)
        return torch.where(x >= 0, x, (1 - delta) / (1 + delta))

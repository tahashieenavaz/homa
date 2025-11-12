import torch
from .ActivationFunction import ActivationFunction


class REU(ActivationFunction):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.where(x >= 0, x, x * torch.exp(x))

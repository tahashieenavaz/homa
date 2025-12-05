import torch
from .ActivationFunction import ActivationFunction


class PLAF(ActivationFunction):
    def __init__(self, d: float = 2):
        super().__init__()
        self.d = d

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        delta = 1 - (1 / self.d)
        sigma = (1 / self.d) * torch.abs(x).pow(self.d)
        return torch.where(x >= 1, x - delta, torch.where(x < -1, -x - delta, sigma))

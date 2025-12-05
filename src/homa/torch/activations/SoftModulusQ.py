import torch
from .ActivationFunction import ActivationFunction


class SoftModulusQ(ActivationFunction):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.where(torch.abs(x) <= 1, x.pow(2) * (2 - x.abs()), x.abs())

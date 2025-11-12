import torch
from .ActivationFunction import ActivationFunction


class DoubleSiLU(ActivationFunction):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a = 1 + torch.exp(-x)
        b = -x * 1 / a
        c = 1 + torch.exp(b)
        return x * 1 / c

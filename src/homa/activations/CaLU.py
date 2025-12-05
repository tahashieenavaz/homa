import torch
import math
from .ActivationFunction import ActivationFunction


class CaLU(ActivationFunction):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a = torch.arctan(x) / math.pi
        b = 0.5
        return x * (a + b)

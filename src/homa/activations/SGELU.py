import torch
import math
from .ActivationFunction import ActivationFunction


class SGELU(ActivationFunction):
    def __init__(self, alpha: float = 0.1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.alpha = alpha

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.alpha * x * torch.erf(x / math.sqrt(2))

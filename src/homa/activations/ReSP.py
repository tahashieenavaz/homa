import torch
import math
from .ActivationFunction import ActivationFunction


class ReSP(ActivationFunction):
    def __init__(self, alpha: float = 1.5):
        super().__init__()
        self.alpha = alpha

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.where(
            x >= 0, self.alpha * x + math.log(2), torch.log(1 + torch.exp(x))
        )

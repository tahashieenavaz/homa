import torch
from .ActivationFunction import ActivationFunction


class AbsLU(ActivationFunction):
    def __init__(self, alpha: float = 0.5):
        super().__init__()
        self.alpha = alpha

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.where(x >= 0, x, torch.abs(x) * self.alpha)

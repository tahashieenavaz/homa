import torch
from .ActivationFunction import ActivationFunction


class DRLU(ActivationFunction):
    def __init__(self, alpha: float = 0.08):
        super().__init__()
        self.alpha = alpha

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.where(x - self.alpha >= 0, x - self.alpha, 0)

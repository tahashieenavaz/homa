import torch
from .ActivationFunction import ActivationFunction


class PoLU(ActivationFunction):
    def __init__(self, alpha: float = 1.5):
        super().__init__()
        if alpha not in [1, 1.5, 2]:
            raise ValueError(f"PoLU alpha must be in [1, 2, 1.5]. Got {alpha}.")

        self.alpha = alpha

    def forward(self, x: torch.Tensor):
        delta = 1 - x
        return torch.where(x >= 0, x, delta.pow(-self.alpha) - 1)

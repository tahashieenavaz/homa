import torch
from .ActivationFunction import ActivationFunction


class SlopedReLU(ActivationFunction):
    def __init__(self, alpha: float = 5.0):
        super().__init__()
        if not (1 <= alpha < 10):
            raise Exception("Sloped ReLU slope needs to be in [1, 10].")
        self.alpha: float = alpha

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.where(x >= 0, self.alpha * x, 0)

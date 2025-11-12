import torch
from .ActivationFunction import ActivationFunction


class SoftModulusT(ActivationFunction):
    def __init__(self, alpha: float = 0.01):
        super().__init__()
        self.alpha = alpha

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.tanh(x / self.alpha)

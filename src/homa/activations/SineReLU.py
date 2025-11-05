import torch
from .ActivationFunction import ActivationFunction


class SineReLU(ActivationFunction):
    def __init__(self, epsilon: float = 0.0025):
        super().__init__()
        self.epsilon: float = epsilon

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.where(x >= 0, x, self.epsilon * (torch.sin(x) - torch.cos(x)))

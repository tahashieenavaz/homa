import torch
from .ActivationFunction import ActivationFunction


class RePU(ActivationFunction):
    def __init__(self, alpha: float = 1.0):
        super().__init__()
        self.alpha = alpha

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.relu(x.pow(self.alpha))

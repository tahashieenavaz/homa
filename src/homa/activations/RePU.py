import torch
from .ActivationFunction import ActivationFunction


class RePU(ActivationFunction):
    def __init__(self, alpha: float):
        super().__init__()
        self.alpha = alpha

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.functional.relu(x.pow(self.alpha))

import torch
from .ActivationFunction import ActivationFunction


class OAF(ActivationFunction):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.relu(x) + x * torch.sigmoid(x)

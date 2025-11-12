import torch
from .ActivationFunction import ActivationFunction


class mReLU(ActivationFunction):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.min(
            torch.nn.functional.relu(1 - x),
            torch.nn.functional.relu(1 + x),
        )

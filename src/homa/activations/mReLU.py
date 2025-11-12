import torch
from .ActivationFunction import ActivationFunction


class mReLU(ActivationFunction):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.min(
            torch.nn.functional.relu(x - 1),
            torch.nn.functional.relu(x + 1),
        )

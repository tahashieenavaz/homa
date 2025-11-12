import torch
from .ActivationFunction import ActivationFunction


class FlattedTSwish(ActivationFunction):
    def __init__(self, t: float = -0.2):
        super().__init__()
        self.t = t

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.relu(x) * torch.sigmoid(x) + self.t

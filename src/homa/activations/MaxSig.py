import torch
from .ActivationFunction import ActivationFunction


class MaxSig(ActivationFunction):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x: torch.Tensor):
        return torch.max(x, torch.sigmoid(x))

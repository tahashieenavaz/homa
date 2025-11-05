import torch
from .ActivationFunction import ActivationFunction


class LogSigmoid(ActivationFunction):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor):
        return torch.log(torch.sigmoid(x))

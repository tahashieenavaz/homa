import torch
from .ActivationFunction import ActivationFunction


class Gish(ActivationFunction):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        a = -torch.exp(x)
        return x * torch.log(2 - torch.exp(a))

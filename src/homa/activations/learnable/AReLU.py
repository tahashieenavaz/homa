import torch
from ..AdaptiveActivationFunction import AdaptiveActivationFunction
from ...device import get_device


class AReLU(AdaptiveActivationFunction):
    def __init__(self):
        super(AReLU, self).__init__()
        self.a = torch.nn.Parameter(torch.tensor(0.9, requires_grad=True))
        self.b = torch.nn.Parameter(torch.tensor(2.0, requires_grad=True))
        self.a.to(get_device())
        self.b.to(get_device())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        negative_slope = torch.clamp(self.a, 0.01, 0.99)
        positive_slope = 1 + torch.sigmoid(self.b)
        positive = positive_slope * torch.relu(x)
        negative = negative_slope * (-torch.relu(-x))
        return positive + negative

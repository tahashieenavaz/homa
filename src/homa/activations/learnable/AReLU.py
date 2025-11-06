import torch
from ..AdaptiveActivationFunction import AdaptiveActivationFunction


class AReLU(AdaptiveActivationFunction):
    def __init__(self):
        super(AReLU, self).__init__()
        self.a = torch.nn.Parameter(torch.tensor(0.9, requires_grad=True))
        self.b = torch.nn.Parameter(torch.tensor(2.0, requires_grad=True))

    def forward(self, z):
        negative_slope = torch.clamp(self.a, 0.01, 0.99)
        positive_slope = 1 + torch.sigmoid(self.b)
        positive = positive_slope * torch.relu(z)
        negative = negative_slope * (-torch.relu(-z))
        return positive + negative

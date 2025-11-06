import torch
from ..AdaptiveActivationFunction import AdaptiveActivationFunction


class PERU(AdaptiveActivationFunction):
    def __init__(self):
        super().__init__()
        self.a = None
        self.b = None
        self.num_channels = None
        self._initialized = False

    def initialize(self, x: torch.Tensor):
        if self._initialized:
            return
        self.num_channels = x.shape[1]
        self.a = torch.nn.Parameter(torch.ones(self.num_channels, requires_grad=True))
        self.b = torch.nn.Parameter(torch.ones(self.num_channels, requires_grad=True))
        self._initialized = True

    def forward(self, x: torch.Tensor):
        self.initialize(x)
        param_shape = (1, self.num_channels) + (1,) * (x.ndim - 2)
        a = self.a.view(param_shape)
        b = self.b.view(param_shape)
        return torch.where(x >= 0, a * x, a * x * torch.exp(b * x))

    def __repr__(self):
        return f"PERU(channels={self.num_channels})"

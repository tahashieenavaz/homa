import torch
from ...device import get_device


class APLU(torch.nn.Module):
    def __init__(self, max_input: float = 1.0):
        super(APLU, self).__init__()
        self.max_input = max_input
        self.alpha = None
        self.beta = None
        self.gamma = None
        self.xi = None
        self.psi = None
        self.mu = None
        self._num_channels = None
        self.device = get_device()

    def _initialize_parameters(self, x):
        if x.ndim < 2:
            raise ValueError(
                f"Input tensor must have at least 2 dimensions (N, C), but got shape {x.shape}"
            )

        num_channels = x.shape[1]
        self._num_channels = num_channels

        param_shape = [1] * x.ndim
        param_shape[1] = num_channels

        self.alpha = torch.nn.Parameter(torch.zeros(param_shape), device=device())
        self.beta = torch.nn.Parameter(torch.zeros(param_shape), device=device())
        self.gamma = torch.nn.Parameter(torch.zeros(param_shape), device=device())

        self.xi = torch.nn.Parameter(self.max_input * torch.rand(param_shape)).to(
            self.device
        )
        self.psi = torch.nn.Parameter(self.max_input * torch.rand(param_shape)).to(
            self.device
        )
        self.mu = torch.nn.Parameter(self.max_input * torch.rand(param_shape)).to(
            self.device
        )

    def forward(self, x):
        if self.alpha is None:
            self._initialize_parameters(x)
        a = torch.relu(x)

        # following are called hinges
        b = self.alpha * torch.relu(-x + self.xi)
        c = self.beta * torch.relu(-x + self.psi)
        d = self.gamma * torch.relu(-x + self.mu)
        z = a + b + c + d
        return z

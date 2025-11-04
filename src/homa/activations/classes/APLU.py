import torch
from torch import nn


class APLU(nn.Module):
    def __init__(self, channels: int, max_input: float = 1.0):
        super().__init__()
        if channels <= 0:
            raise ValueError(f"Number of channels must be positive, got {channels}.")
        self.channels = int(channels)
        self.max_input = float(max_input)
        self.alpha = nn.Parameter(torch.empty(self.channels))
        self.beta = nn.Parameter(torch.empty(self.channels))
        self.gamma = nn.Parameter(torch.empty(self.channels))
        self.xi = nn.Parameter(torch.empty(self.channels))
        self.psi = nn.Parameter(torch.empty(self.channels))
        self.mu = nn.Parameter(torch.empty(self.channels))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        with torch.no_grad():
            self.alpha.zero_()
            self.beta.zero_()
            self.gamma.zero_()
            self.xi.uniform_(0.0, self.max_input)
            self.psi.uniform_(0.0, self.max_input)
            self.mu.uniform_(0.0, self.max_input)

    @staticmethod
    def _reshape_for_broadcast(param: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        return param.view(1, param.shape[0], *([1] * (ref.ndim - 2)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim < 2:
            raise ValueError(
                f"APLU expects inputs with at least two dimensions (N, C, ...), got {tuple(x.shape)}."
            )
        if int(x.shape[1]) != self.channels:
            raise ValueError(
                f"APLU was initialized with {self.channels} channels but received input with {int(x.shape[1])}."
            )

        alpha = self._reshape_for_broadcast(self.alpha, x)
        beta = self._reshape_for_broadcast(self.beta, x)
        gamma = self._reshape_for_broadcast(self.gamma, x)
        xi = self._reshape_for_broadcast(self.xi, x)
        psi = self._reshape_for_broadcast(self.psi, x)
        mu = self._reshape_for_broadcast(self.mu, x)
        output = torch.relu(x)
        output = output + alpha * torch.relu(-x + xi)
        output = output + beta * torch.relu(-x + psi)
        output = output + gamma * torch.relu(-x + mu)
        return output

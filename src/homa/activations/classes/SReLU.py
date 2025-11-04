import torch
from torch import nn


class SReLU(nn.Module):
    def __init__(
        self,
        channels: int,
        alpha_init: float = 0.0,
        beta_init: float = 0.0,
        gamma_init: float = 1.0,
        delta_init: float = 1.0,
    ):
        super().__init__()
        if channels <= 0:
            raise ValueError(f"channels must be positive, got {channels}.")
        self.channels = int(channels)
        self.alpha_init_val = float(alpha_init)
        self.beta_init_val = float(beta_init)
        self.gamma_init_val = float(gamma_init)
        self.delta_init_val = float(delta_init)
        shape = (1, self.channels, 1, 1)
        self.alpha = nn.Parameter(torch.empty(shape))
        self.beta = nn.Parameter(torch.empty(shape))
        self.gamma = nn.Parameter(torch.empty(shape))
        self.delta = nn.Parameter(torch.empty(shape))
        self.reset_parameters()

    def reset_parameters(self):
        with torch.no_grad():
            self.alpha.fill_(self.alpha_init_val)
            self.beta.fill_(self.beta_init_val)
            self.gamma.fill_(self.gamma_init_val)
            self.delta.fill_(self.delta_init_val)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 4:
            raise ValueError(
                f"Expected 4D input (N, C, H, W), got {x.dim()}D with shape {tuple(x.shape)}"
            )
        if int(x.shape[1]) != self.channels:
            raise ValueError(
                f"SReLU was initialized with C={self.channels} but received input with C={int(x.shape[1])}."
            )

        start = self.beta + self.alpha * (x - self.beta)
        finish = self.delta + self.gamma * (x - self.delta)
        out = torch.where(x < self.beta, start, x)
        out = torch.where(x > self.delta, finish, out)
        return out

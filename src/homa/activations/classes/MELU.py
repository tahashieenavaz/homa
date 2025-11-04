import torch
from torch import nn
import torch.nn.functional as F


class MELU(nn.Module):
    def __init__(self, channels: int, max_input: float = 1.0):
        super().__init__()
        if channels <= 0:
            raise ValueError(f"channels must be positive, got {channels}.")
        self.channels = int(channels)
        self.max_input = float(max_input)
        shape = (1, self.channels, 1, 1)
        self.alpha = nn.Parameter(torch.empty(shape))
        self.beta = nn.Parameter(torch.empty(shape))
        self.gamma = nn.Parameter(torch.empty(shape))
        self.delta = nn.Parameter(torch.empty(shape))
        self.xi = nn.Parameter(torch.empty(shape))
        self.psi = nn.Parameter(torch.empty(shape))
        self.reset_parameters()

    def reset_parameters(self):
        with torch.no_grad():
            for param in (
                self.alpha,
                self.beta,
                self.gamma,
                self.delta,
                self.xi,
                self.psi,
            ):
                param.zero_()

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        if X.dim() != 4:
            raise ValueError(
                f"Expected 4D input (N, C, H, W), got {X.dim()}D with shape {tuple(X.shape)}"
            )
        if int(X.shape[1]) != self.channels:
            raise ValueError(
                f"MELU was initialized with C={self.channels} but received input with C={int(X.shape[1])}."
            )

        X_norm = X / self.max_input
        Y = torch.roll(X_norm, shifts=-1, dims=1)

        term1 = F.relu(X_norm)
        term2 = self.alpha * torch.clamp(X_norm, max=0)

        dist_sq_beta = (X_norm - 2) ** 2 + (Y - 2) ** 2
        dist_sq_gamma = (X_norm - 1) ** 2 + (Y - 1) ** 2
        dist_sq_delta = (X_norm - 1) ** 2 + (Y - 3) ** 2
        dist_sq_xi = (X_norm - 3) ** 2 + (Y - 1) ** 2
        dist_sq_psi = (X_norm - 3) ** 2 + (Y - 3) ** 2

        term3 = self.beta * torch.sqrt(F.relu(2 - dist_sq_beta))
        term4 = self.gamma * torch.sqrt(F.relu(1 - dist_sq_gamma))
        term5 = self.delta * torch.sqrt(F.relu(1 - dist_sq_delta))
        term6 = self.xi * torch.sqrt(F.relu(1 - dist_sq_xi))
        term7 = self.psi * torch.sqrt(F.relu(1 - dist_sq_psi))

        Z_norm = term1 + term2 + term3 + term4 + term5 + term6 + term7
        return Z_norm * self.max_input

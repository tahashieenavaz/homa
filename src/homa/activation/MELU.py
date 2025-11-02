import torch
from .utils import (
    as_channel_parameters,
    negative_part,
    positive_part,
    device_compatibility_check,
)


class GALU(torch.nn.Module):
    def __init__(self, channels: int, max_input: float):
        super().__init__()
        self.channels = int(channels)
        self.max_input = float(max_input)

        self.alpha1 = torch.nn.Parameter(torch.zeros(self.channels))
        self.alpha2 = torch.nn.Parameter(torch.zeros(self.channels))
        self.beta1 = torch.nn.Parameter(torch.zeros(self.channels))
        self.beta2 = torch.nn.Parameter(torch.zeros(self.channels))
        self.gamma1 = torch.nn.Parameter(torch.zeros(self.channels))
        self.gamma2 = torch.nn.Parameter(torch.zeros(self.channels))
        self.delta1 = torch.nn.Parameter(torch.zeros(self.channels))
        self.delta2 = torch.nn.Parameter(torch.zeros(self.channels))
        self.c1 = torch.nn.Parameter(torch.full((self.channels,), 0.5))
        self.c2 = torch.nn.Parameter(torch.full((self.channels,), 0.5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_shape = x.shape

        # Normalize to 4D (N,C,H,W) for uniform broadcasting
        if x.dim() == 1:
            # (C,) -> (1,C,1,1)
            x = x.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        elif x.dim() == 2:
            # (N,C) -> (N,C,1,1)
            x = x.unsqueeze(-1).unsqueeze(-1)
        elif x.dim() == 4:
            pass
        else:
            raise ValueError(
                f"Unsupported input dimensions {x.dim()}; expected 1D, 2D, or 4D."
            )

        device_compatibility_check(self, x)

        X = x / self.max_input

        A1 = as_channel_parameters(self.alpha1, X)
        B1 = as_channel_parameters(self.beta1, X)
        G1 = as_channel_parameters(self.gamma1, X)
        D1 = as_channel_parameters(self.delta1, X)
        A2 = as_channel_parameters(self.alpha2, X)
        B2 = as_channel_parameters(self.beta2, X)
        G2 = as_channel_parameters(self.gamma2, X)
        D2 = as_channel_parameters(self.delta2, X)
        C1 = as_channel_parameters(self.c1, X)
        C2 = as_channel_parameters(self.c2, X)

        Z1 = positive_part(X) + A1 * negative_part(X)
        Z1 = Z1 + B1 * (
            positive_part(1 - torch.abs(X - 1))
            + torch.minimum(torch.abs(X - 3) - 1, torch.zeros_like(X))
        )
        Z1 = Z1 + G1 * (
            positive_part(0.5 - torch.abs(X - 0.5))
            + torch.minimum(torch.abs(X - 1.5) - 0.5, torch.zeros_like(X))
        )
        Z1 = Z1 + D1 * (
            positive_part(0.5 - torch.abs(X - 2.5))
            + torch.minimum(torch.abs(X - 3.5) - 0.5, torch.zeros_like(X))
        )
        Z1 = self.max_input * Z1
        Z2 = (
            positive_part(X)
            + A2 * negative_part(X)
            + B2 * torch.minimum(torch.abs(X - 2) - 2, torch.zeros_like(X))
            + G2 * positive_part(-torch.abs(X - 1) + 1)
            + D2 * positive_part(-torch.abs(X - 3) + 1)
        )
        Z2 = self.max_input * Z2
        Z = C1 * Z1 + C2 * Z2

        if len(orig_shape) == 1:
            Z = Z.squeeze(-1).squeeze(-1).squeeze(0)  # back to (C,)
        elif len(orig_shape) == 2:
            Z = Z.squeeze(-1).squeeze(-1)  # back to (N,C)

        return Z

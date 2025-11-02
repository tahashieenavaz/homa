import torch
from .utils import positive_part, negative_part, as_channel_parameters


class WideMELU(torch.nn.Module):
    def __init__(self, channels: int, max_input: float):
        super().__init__()
        self.channels = int(channels)
        self.max_input = float(max_input)

        self.alpha = torch.nn.Parameter(torch.zeros(self.channels))
        self.coeffs = torch.nn.Parameter(torch.zeros(self.channels, 7))
        a_vals = torch.tensor([2.0, 1.0, 3.0, 0.5, 1.5, 2.5, 3.5])
        lam_vals = torch.tensor([2.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.5])
        self.register_buffer("_a", a_vals.view(1, 1, 1, 1, 7))
        self.register_buffer("_lam", lam_vals.view(1, 1, 1, 1, 7))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig = x.shape

        if x.dim() == 1:  # (C,)
            x = x.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        elif x.dim() == 2:  # (N,C)
            x = x.unsqueeze(-1).unsqueeze(-1)
        elif x.dim() != 4:  # (N,C,H,W) expected otherwise
            raise ValueError(f"Expected 1D, 2D, or 4D input, got rank {x.dim()}")

        X = x / self.max_input

        A = as_channel_parameters(self.alpha, X)
        Z = positive_part(X) + A * negative_part(X)

        X5 = X.unsqueeze(-1)  # (N,C,H,W,1)
        a5 = self._a.to(dtype=X.dtype, device=X.device)
        lam5 = self._lam.to(dtype=X.dtype, device=X.device)
        phi_all = self.phi_hat(X5, a5, lam5)  # (N,C,H,W,7)

        coeffs = self.coeffs.to(dtype=X.dtype, device=X.device).view(1, -1, 1, 1, 7)
        Z = Z + (phi_all * coeffs).sum(dim=-1)

        Z = self.max_input * Z

        if len(orig) == 1:
            Z = Z.squeeze(-1).squeeze(-1).squeeze(0)
        elif len(orig) == 2:
            Z = Z.squeeze(-1).squeeze(-1)
        return Z

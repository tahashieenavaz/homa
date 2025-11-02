import torch


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

    @staticmethod
    def positive_part(x):
        return torch.maximum(x, torch.zeros_like(x))

    @staticmethod
    def negative_part(x):
        return torch.minimum(x, torch.zeros_like(x))

    @staticmethod
    def phi_hat(X5, a5, lam5):
        term_pos = torch.maximum(lam5 - torch.abs(X5 - a5), torch.zeros_like(X5))
        term_neg = torch.minimum(
            torch.abs(X5 - (a5 + 2 * lam5)) - lam5, torch.zeros_like(X5)
        )
        return term_pos + term_neg

    def as_channel_parameters(self, p, x):
        shape = [1] * x.dim()
        shape[1] = -1
        return p.view(*shape)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig = x.shape

        if x.dim() == 1:  # (C,)
            x = x.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        elif x.dim() == 2:  # (N,C)
            x = x.unsqueeze(-1).unsqueeze(-1)
        elif x.dim() != 4:  # (N,C,H,W) expected otherwise
            raise ValueError(f"Expected 1D, 2D, or 4D input, got rank {x.dim()}")

        X = x / self.max_input

        A = self.as_channel_parameters(self.alpha, X)
        Z = self.positive_part(X) + A * self.negative_part(X)

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

import torch


class Galu(torch.nn.Module):
    def __init__(self, channels: int, max_input: float):
        super().__init__()
        self.channels = int(channels)
        self.max_input = float(max_input)

        self.alpha = torch.nn.Parameter(torch.zeros(self.channels))
        self.beta = torch.nn.Parameter(torch.zeros(self.channels))
        self.gamma = torch.nn.Parameter(torch.zeros(self.channels))
        self.delta = torch.nn.Parameter(torch.zeros(self.channels))

    @staticmethod
    def positive_part(x):
        return torch.maximum(x, torch.zeros_like(x))

    @staticmethod
    def negative_part(x):
        return torch.minimum(x, torch.zeros_like(x))

    def as_channel_parameters(self, p: torch.Tensor, x: torch.Tensor):
        shape = [1] * x.dim()
        shape[1] = -1
        return p.view(*shape)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_shape = x.shape

        # Normalize shapes to (N,C,H,W) for clean broadcasting
        if x.dim() == 1:  # (C,)
            x = x.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        elif x.dim() == 2:  # (N,C)
            x = x.unsqueeze(-1).unsqueeze(-1)
        elif x.dim() == 4:  # (N,C,H,W)
            pass
        else:
            raise ValueError(
                f"Unsupported input rank {x.dim()}; expected 1D, 2D, or 4D."
            )

        X = x / self.max_input

        A = self._as_channel_param(self.alpha, X)
        B = self._as_channel_param(self.beta, X)
        G = self._as_channel_param(self.gamma, X)
        D = self._as_channel_param(self.delta, X)

        Z = self.positive_part(X) + A * self.negative_part(X)  # PReLU-like
        Z = Z + B * (
            self.positive_part(1 - torch.abs(X - 1))
            + torch.minimum(torch.abs(X - 3) - 1, torch.zeros_like(X))
        )
        Z = Z + G * (
            self.positive_part(0.5 - torch.abs(X - 0.5))
            + torch.minimum(torch.abs(X - 1.5) - 0.5, torch.zeros_like(X))
        )
        Z = Z + D * (
            self.positive_part(0.5 - torch.abs(X - 2.5))
            + torch.minimum(torch.abs(X - 3.5) - 0.5, torch.zeros_like(X))
        )
        Z = self.max_input * Z

        if len(orig_shape) == 1:
            Z = Z.squeeze(-1).squeeze(-1).squeeze(0)  # (C,)
        elif len(orig_shape) == 2:
            Z = Z.squeeze(-1).squeeze(-1)  # (N,C)

        return Z

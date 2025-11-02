import torch


class SmallGALU(torch.nn.Module):
    def __init__(self, channels: int, max_input: float):
        super().__init__()
        self.channels = int(channels)
        self.max_input = float(max_input)
        self.alpha = torch.nn.Parameter(torch.zeros(self.channels))
        self.beta = torch.nn.Parameter(torch.zeros(self.channels))

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
        # normalize to NCHW for clean broadcasting
        if x.dim() == 1:  # (C,)
            x = x.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        elif x.dim() == 2:  # (N,C)
            x = x.unsqueeze(-1).unsqueeze(-1)
        elif x.dim() != 4:
            raise ValueError(f"Expected 1D, 2D, or 4D input, got rank {x.dim()}")

        X = x / self.max_input
        A = self.as_channel_parameters(self.alpha, X)
        B = self.as_channel_parameters(self.beta, X)

        Z = self.positive_part(X) + A * self.negative_part(X)  # PReLU core

        Z = Z + B * (
            self.positive_part(1 - torch.abs(X - 1))
            + torch.minimum(torch.abs(X - 3) - 1, torch.zeros_like(X))
        )

        Z = self.max_input * Z

        if len(orig_shape) == 1:
            Z = Z.squeeze(-1).squeeze(-1).squeeze(0)
        elif len(orig_shape) == 2:
            Z = Z.squeeze(-1).squeeze(-1)
        return Z

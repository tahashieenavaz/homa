import torch


class APLU(torch.nn.Module):
    def __init__(
        self, channels: int | None = None, n: int = 2, init_b: str = "linspace"
    ):
        super().__init__()
        self.n = n
        self.init_b = init_b
        if channels is None:
            self.register_parameter("a", None)
            self.register_parameter("b", None)
        else:
            self._init_params(channels, device=None, dtype=None)

    def _init_params(self, channels, device, dtype):
        a = torch.zeros(channels, self.n, device=device, dtype=dtype)
        if self.init_b == "linspace":
            b = (
                torch.linspace(-1.0, 1.0, steps=self.n, device=device, dtype=dtype)
                .expand(channels, -1)
                .contiguous()
            )
        else:
            b = torch.empty(channels, self.n, device=device, dtype=dtype).uniform_(
                -1.0, 1.0
            )
        self.a = torch.nn.Parameter(a)
        self.b = torch.nn.Parameter(b)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.a is None or self.b is None:
            self._init_params(x.shape[1], device=x.device, dtype=x.dtype)

        y = F.relu(x)
        x_exp = x.unsqueeze(-1)
        expand_shape = (
            (
                1,
                x.shape[1],
            )
            + (1,) * (x.dim() - 2)
            + (self.n,)
        )
        a = self.a.view(*expand_shape)
        b = self.b.view(*expand_shape)
        hinges = (-x_exp + b).clamp_max(0.0)
        return y + (a * hinges).sum(dim=-1)

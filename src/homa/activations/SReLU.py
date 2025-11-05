import torch


class SReLU(torch.nn.Module):
    def __init__(self, channels: int | None = None, max_input: float = 1.0):
        super().__init__()
        self.M = float(max_input)
        if channels is None:
            self.register_parameter("t_l", None)
            self.register_parameter("t_r", None)
            self.register_parameter("a_l", None)
            self.register_parameter("a_r", None)
        else:
            self._init_params(channels, None, None)

    def _init_params(self, C: int, device, dtype):
        self.t_l = torch.nn.Parameter(torch.zeros(C, device=device, dtype=dtype))
        self.t_r = torch.nn.Parameter(
            torch.full((C,), self.M, device=device, dtype=dtype)
        )
        self.a_l = torch.nn.Parameter(torch.zeros(C, device=device, dtype=dtype))
        self.a_r = torch.nn.Parameter(torch.ones(C, device=device, dtype=dtype))

    def _expand_param(p: torch.Tensor, x: torch.Tensor):
        return p.view((1, x.shape[1]) + (1,) * (x.dim() - 2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.t_l is None:
            self._init_params(x.shape[1], x.device, x.dtype)

        t_l = self._expand_param(self.t_l, x)
        t_r = self._expand_param(self.t_r, x)
        a_l = self._expand_param(self.a_l, x)
        a_r = self._expand_param(self.a_r, x)
        y = torch.where(x < t_l, t_l + a_l * (x - t_l), x)
        y = torch.where(x > t_r, t_r + a_r * (x - t_r), y)
        return y

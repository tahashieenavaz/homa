import torch


class SReLU(torch.nn.Module):
    def __init__(
        self,
        alpha_init: float = 0.0,
        beta_init: float = 0.0,
        gamma_init: float = 1.0,
        delta_init: float = 1.0,
    ):
        super().__init__()
        self.alpha_init_val = alpha_init
        self.beta_init_val = beta_init
        self.gamma_init_val = gamma_init
        self.delta_init_val = delta_init
        self.alpha = torch.nn.UninitializedParameter()
        self.beta = torch.nn.UninitializedParameter()
        self.gamma = torch.nn.UninitializedParameter()
        self.delta = torch.nn.UninitializedParameter()

    def _initialize_parameters(self, x: torch.Tensor):
        if isinstance(self.alpha, torch.nn.UninitializedParameter):
            if x.dim() < 2:
                raise ValueError(
                    f"Input tensor must have at least 2 dimensions (N, C), but got {x.dim()}"
                )

            num_channels = x.shape[1]
            param_shape = [1] * x.dim()
            param_shape[1] = num_channels
            self.alpha = torch.nn.Parameter(
                torch.full(param_shape, self.alpha_init_val)
            )
            self.beta = torch.nn.Parameter(torch.full(param_shape, self.beta_init_val))
            self.gamma = torch.nn.Parameter(
                torch.full(param_shape, self.gamma_init_val)
            )
            self.delta = torch.nn.Parameter(
                torch.full(param_shape, self.delta_init_val)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._initialize_parameters(x)
        start = self.beta + self.alpha * (x - self.beta)
        finish = self.delta + self.gamma * (x - self.delta)
        out = torch.where(x < self.beta, start, x)
        out = torch.where(x > self.delta, finish, out)
        return out

import torch
from .ActivationFunction import ActivationFunction


class BaseDLReLU(ActivationFunction):
    def __init__(self, a: float = 0.01, init_mse: float = 1.0, mode: str = "linear"):
        super().__init__()
        assert 0.0 < a < 1.0, "a must be in (0,1)"
        assert mode in ("linear", "exp")
        self.a = float(a)
        self.mode = mode
        self.register_buffer("prev_mse", torch.tensor(float(init_mse)))

    @torch.no_grad()
    def set_prev_mse(self, mse_value):
        if isinstance(mse_value, torch.Tensor):
            mse_value = float(mse_value.detach().cpu().item())
        self.prev_mse.fill_(mse_value)

    @torch.no_grad()
    def update_from_loss(self, loss_tensor: torch.Tensor):
        self.set_prev_mse(loss_tensor)

    def forward(self, x: torch.Tensor, mse_prev: torch.Tensor | float | None = None):
        b_t = (
            self.prev_mse
            if mse_prev is None
            else (torch.as_tensor(mse_prev, device=x.device, dtype=x.dtype))
        )
        if self.mode == "linear":
            slope = self.a * b_t
        else:
            slope = self.a * torch.exp(-b_t)
        return torch.where(x >= 0, x, slope * x)

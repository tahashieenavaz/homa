import torch


class PDELU(torch.nn.Module):
    def __init__(self, theta: float = 0.5):
        super(PDELU, self).__init__()
        if theta == 1.0:
            raise ValueError(
                "theta cannot be 1.0, as it would cause a division by zero."
            )
        self.theta = theta
        self._power_val = 1.0 / (1.0 - self.theta)
        self.alpha = torch.nn.UninitializedParameter()
        self._num_channels = None

    def _initialize_parameters(self, x: torch.Tensor):
        if x.ndim < 2:
            raise ValueError(
                f"Input tensor must have at least 2 dimensions (N, C), but got shape {x.shape}"
            )

        num_channels = x.shape[1]
        self._num_channels = num_channels
        param_shape = [1] * x.ndim
        param_shape[1] = num_channels
        init_tensor = torch.zeros(param_shape) + 0.1
        self.alpha = torch.nn.Parameter(init_tensor)

    def forward(self, x: torch.Tensor):
        if self.alpha is None:
            self._initialize_parameters(x)

        zero = torch.tensor(0.0, device=x.device, dtype=x.dtype)
        positive_part = torch.relu(x)
        inner_term = torch.relu(1.0 + (1.0 - self.theta) * x)
        powered_term = torch.pow(inner_term, self._power_val)
        subtracted_term = powered_term - 1.0
        negative_part = self.alpha * torch.min(subtracted_term, zero)
        return positive_part + negative_part

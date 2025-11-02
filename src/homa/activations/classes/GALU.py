import torch


class GALU(torch.nn.Module):
    def __init__(self, max_input: float = 1.0):
        super(GALU, self).__init__()
        if max_input <= 0:
            raise ValueError("max_input must be positive.")
        self.max_input = max_input
        self.alpha = None
        self.beta = None
        self.gamma = None
        self.delta = None
        self._num_channels = None

    def _initialize_parameters(self, x):
        if x.ndim < 2:
            raise ValueError(
                f"Input tensor must have at least 2 dimensions (N, C), but got shape {x.shape}"
            )

        num_channels = x.shape[1]
        self._num_channels = num_channels
        param_shape = [1] * x.ndim
        param_shape[1] = num_channels
        self.alpha = torch.nn.Parameter(torch.zeros(param_shape))
        self.beta = torch.nn.Parameter(torch.zeros(param_shape))
        self.gamma = torch.nn.Parameter(torch.zeros(param_shape))
        self.delta = torch.nn.Parameter(torch.zeros(param_shape))

    def forward(self, x):
        if self.alpha is None:
            self._initialize_parameters(x)

        zero = torch.tensor(0.0, device=x.device, dtype=x.dtype)
        x_norm = x / self.max_input
        part_prelu = torch.relu(x_norm) + self.alpha * torch.min(x_norm, zero)
        part_beta = self.beta * (
            torch.relu(1.0 - torch.abs(x_norm - 1.0))
            + torch.min(torch.abs(x_norm - 3.0) - 1.0, zero)
        )
        part_gamma = self.gamma * (
            torch.relu(0.5 - torch.abs(x_norm - 0.5))
            + torch.min(torch.abs(x_norm - 1.5) - 0.5, zero)
        )
        part_delta = self.delta * (
            torch.relu(0.5 - torch.abs(x_norm - 2.5))
            + torch.min(torch.abs(x_norm - 3.5) - 0.5, zero)
        )
        z = part_prelu + part_beta + part_gamma + part_delta
        return z * self.max_input

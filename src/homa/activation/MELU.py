import torch


class MELU(torch.nn.Module):
    def __init__(self, max_input=1.0):
        super(MELU, self).__init__()
        if max_input <= 0:
            raise ValueError("max_input must be positive.")
        self.max_input = max_input
        self.alpha1 = None
        self.alpha2 = None
        self.beta1 = None
        self.beta2 = None
        self.gamma1 = None
        self.gamma2 = None
        self.delta1 = None
        self.delta2 = None
        self.c1 = None
        self.c2 = None
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

        self.alpha1 = torch.nn.Parameter(torch.zeros(param_shape))
        self.beta1 = torch.nn.Parameter(torch.zeros(param_shape))
        self.gamma1 = torch.nn.Parameter(torch.zeros(param_shape))
        self.delta1 = torch.nn.Parameter(torch.zeros(param_shape))
        self.alpha2 = torch.nn.Parameter(torch.zeros(param_shape))
        self.beta2 = torch.nn.Parameter(torch.zeros(param_shape))
        self.gamma2 = torch.nn.Parameter(torch.zeros(param_shape))
        self.delta2 = torch.nn.Parameter(torch.zeros(param_shape))
        self.c1 = torch.nn.Parameter(torch.zeros(param_shape) + 0.5)
        self.c2 = torch.nn.Parameter(torch.zeros(param_shape) + 0.5)

    def forward(self, x):
        if self.alpha1 is None:
            self._initialize_parameters(x)

        zero = torch.tensor(0.0, device=x.device, dtype=x.dtype)
        x_norm = x / self.max_input
        z1_part1 = torch.relu(x_norm) + self.alpha1 * torch.min(x_norm, zero)
        z1_part2 = self.beta1 * (
            torch.relu(1.0 - torch.abs(x_norm - 1.0))
            + torch.min(torch.abs(x_norm - 3.0) - 1.0, zero)
        )
        z1_part3 = self.gamma1 * (
            torch.relu(0.5 - torch.abs(x_norm - 0.5))
            + torch.min(torch.abs(x_norm - 1.5) - 0.5, zero)
        )
        z1_part4 = self.delta1 * (
            torch.relu(0.5 - torch.abs(x_norm - 2.5))
            + torch.min(torch.abs(x_norm - 3.5) - 0.5, zero)
        )
        z1_final = self.max_input * (z1_part1 + z1_part2 + z1_part3 + z1_part4)
        z2_part1 = torch.relu(x_norm) + self.alpha2 * torch.min(x_norm, zero)
        z2_part2 = self.beta2 * torch.min(torch.abs(x_norm - 2.0) - 2.0, zero)
        z2_part3 = self.gamma2 * torch.relu(1.0 - torch.abs(x_norm - 1.0))
        z2_part4 = self.delta2 * torch.relu(1.0 - torch.abs(x_norm - 3.0))
        z2_final = self.max_input * (z2_part1 + z2_part2 + z2_part3 + z2_part4)
        z = self.c1 * z1_final + self.c2 * z2_final
        return z

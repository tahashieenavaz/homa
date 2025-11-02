import torch


class WideMELU(torch.nn.Module):
    def __init__(self, maxInput: float = 1.0):
        super().__init__()
        self.maxInput = float(maxInput)
        self.alpha = None
        self.beta = None
        self.gamma = None
        self.delta = None
        self.xi = None
        self.psi = None
        self.theta = None
        self.lam = None
        self._initialized = False

    def _initialize_parameters(self, X: torch.Tensor):
        if X.dim() != 4:
            raise ValueError(
                f"Expected 4D input (B, C, H, W), but got {X.dim()}D input."
            )

        num_channels = X.shape[1]
        shape = (1, num_channels, 1, 1)

        self.alpha = torch.nn.Parameter(torch.zeros(shape))
        self.beta = torch.nn.Parameter(torch.zeros(shape))
        self.gamma = torch.nn.Parameter(torch.zeros(shape))
        self.delta = torch.nn.Parameter(torch.zeros(shape))
        self.xi = torch.nn.Parameter(torch.zeros(shape))
        self.psi = torch.nn.Parameter(torch.zeros(shape))
        self.theta = torch.nn.Parameter(torch.zeros(shape))
        self.lam = torch.nn.Parameter(torch.zeros(shape))
        self._initialized = True

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        if not self._initialized:
            self._initialize_parameters(X)
        X_norm = X / self.maxInput
        Y = torch.roll(X_norm, shifts=-1, dims=1)
        term1 = torch.relu(X_norm)
        term2 = self.alpha * torch.clamp(X_norm, max=0)
        dist_sq_beta = (X_norm - 2) ** 2 + (Y - 2) ** 2
        dist_sq_gamma = (X_norm - 1) ** 2 + (Y - 1) ** 2
        dist_sq_delta = (X_norm - 1) ** 2 + (Y - 3) ** 2
        dist_sq_xi = (X_norm - 3) ** 2 + (Y - 1) ** 2
        dist_sq_psi = (X_norm - 3) ** 2 + (Y - 3) ** 2
        dist_sq_theta = (X_norm - 1) ** 2 + (Y - 2) ** 2
        dist_sq_lambda = (X_norm - 3) ** 2 + (Y - 2) ** 2

        term3 = self.beta * torch.sqrt(torch.relu(2 - dist_sq_beta))
        term4 = self.gamma * torch.sqrt(torch.relu(1 - dist_sq_gamma))
        term5 = self.delta * torch.sqrt(torch.relu(1 - dist_sq_delta))
        term6 = self.xi * torch.sqrt(torch.relu(1 - dist_sq_xi))
        term7 = self.psi * torch.sqrt(torch.relu(1 - dist_sq_psi))
        term8 = self.theta * torch.sqrt(torch.relu(1 - dist_sq_theta))
        term9 = self.lam * torch.sqrt(torch.relu(1 - dist_sq_lambda))
        Z_norm = term1 + term2 + term3 + term4 + term5 + term6 + term7 + term8 + term9
        Z = Z_norm * self.maxInput
        return Z

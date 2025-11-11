import torch


class SoftActorModule(torch.nn.Module):
    def __init__(
        self,
        state_dimension: int,
        hidden_dimension: int,
        action_dimension: int,
        min_std: float,
        max_std: float,
    ):
        super().__init__()

        self.state_dimension: int = state_dimension
        self.hidden_dimension: int = hidden_dimension
        self.action_dimension: int = action_dimension
        self.min_std: float = float(min_std)
        self.max_std: float = float(max_std)

        self.phi = torch.nn.Sequential(
            torch.nn.Linear(self.state_dimension, self.hidden_dimension),
            torch.nn.ReLU(),
            torch.nn.Linear(self.hidden_dimension, self.hidden_dimension),
            torch.nn.ReLU(),
        )

        self.mu = torch.nn.Linear(self.hidden_dimension, self.action_dimension)
        torch.nn.init.uniform_(self.mu.weight, -3e-3, 3e-3)
        torch.nn.init.constant_(self.mu.bias, 0.0)

        self.xi = torch.nn.Linear(self.hidden_dimension, self.action_dimension)
        torch.nn.init.constant_(self.xi.weight, 0.0)
        torch.nn.init.constant_(self.xi.bias, -0.5)

    def forward(self, state: torch.Tensor):
        features = self.phi(state)
        mean = self.mu(features)
        std = self.xi(features).clamp(self.min_std, self.max_std).exp()
        return mean, std

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
        self.xi = torch.nn.Linear(self.hidden_dimension, self.action_dimension)

    def forward(self, state: torch.Tensor):
        features = self.phi(state)
        mean = self.mu(features)
        std = self.xi(features)
        std = std.clamp(self.min_std, self.max_std)
        return mean, std

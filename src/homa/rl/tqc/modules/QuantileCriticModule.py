import torch


class QuantileCriticModule(torch.nn.Module):
    def __init__(
        self,
        state_dimension: int,
        action_dimension: int,
        hidden_dimension: int,
        num_quantiles: int,
    ):
        super().__init__()
        self.phi = torch.nn.Sequential(
            torch.nn.Linear(state_dimension + action_dimension, hidden_dimension),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dimension, hidden_dimension),
            torch.nn.ReLU(),
        )
        self.fc = torch.nn.Linear(hidden_dimension, num_quantiles)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        psi = torch.cat([state, action], dim=-1)
        features = self.phi(psi)
        return self.fc(features)

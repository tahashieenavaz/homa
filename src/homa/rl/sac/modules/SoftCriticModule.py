import torch


class SoftCriticModule(torch.nn.Module):
    def __init__(
        self,
        state_dimension: int,
        hidden_dimension: int,
        action_dimension: int,
    ):
        super().__init__()

        self.state_dimension: int = state_dimension
        self.action_dimension: int = action_dimension
        self.hidden_dimension: int = hidden_dimension

        self.phi = torch.nn.Sequential(
            torch.nn.Linear(
                self.state_dimension + self.action_dimension, self.hidden_dimension
            ),
            torch.nn.ReLU(),
            torch.nn.Linear(self.hidden_dimension, self.hidden_dimension),
            torch.nn.ReLU(),
        )
        self.fc = torch.nn.Linear(self.hidden_dimension, 1)

    def forward(self, state: torch.Tensor, action: torch.Tensor):
        psi = torch.cat([state, action], dim=1)
        features = self.phi(psi)
        return self.fc(features)

import torch


class ContinuousActorModule(torch.nn.Module):
    def __init__(
        self,
        state_dimension: int,
        action_dimension: int,
        hidden_dimension: int,
        num_skills: int,
        min_std: float,
        max_std: float,
    ):
        super().__init__()
        self.state_dimension: int = state_dimension
        self.action_dimension: int = action_dimension
        self.num_skills: int = num_skills
        self.hidden_dimension: int = hidden_dimension
        self.input_dimension: int = self.state_dimension + self.num_skills
        self.min_std: float = min_std
        self.max_std: float = max_std

        self.phi = torch.nn.Sequential(
            torch.nn.Linear(self.input_dimension, self.hidden_dimension),
            torch.nn.ReLU(),
            torch.nn.Linear(self.hidden_dimension, self.hidden_dimension),
            torch.nn.ReLU(),
            torch.nn.Linear(self.hidden_dimension, self.hidden_dimension),
        )

        self.mu = torch.nn.Linear(self.hidden_dimension, self.action_dimension)
        self.xi = torch.nn.Linear(self.hidden_dimension, self.action_dimension)

    def forward(self, state: torch.Tensor, skill: torch.Tensor) -> torch.Tensor:
        # fix the size to be one state per batch
        state = state.view(state.size(0), -1)

        psi = torch.cat([state, skill], dim=-1)
        features = self.phi(psi)
        mean = self.mu(features)
        std = self.xi(features).clamp(self.min_std, self.max_std)
        return mean, std

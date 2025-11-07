import torch
from torch.distributions import Normal


class ContinuousActorModule(torch.nn.Module):
    def __init__(
        self,
        state_dimension: int,
        action_dimension: int,
        hidden_dimension: int,
        num_skills: int,
        epsilon: float,
    ):
        super().__init__()
        self.state_dimension: int = state_dimension
        self.action_dimension: int = action_dimension
        self.num_skills: int = num_skills
        self.hidden_dimension: int = hidden_dimension
        self.epsilon: float = epsilon
        self.input_dimension = self.state_dimension + self.num_skills

        self.phi = torch.nn.Sequential(
            torch.nn.Linear(self.input_dimension, self.hidden_dimension),
            torch.nn.ReLU(),
            torch.nn.Linear(self.hidden_dimension, self.hidden_dimension),
            torch.nn.ReLU(),
            torch.nn.Linear(self.hidden_dimension, self.action_dimension),
        )

        self.mu = torch.nn.Linear(self.hidden_dimension, self.action_dimension)
        self.xi = torch.nn.Linear(self.hidden_dimension, self.action_dimension)

    def forward(self, state: torch.Tensor, skill: torch.Tensor) -> torch.Tensor:
        psi = torch.cat([state, skill], dim=-1)
        features = self.phi(psi)
        mean = self.mu(features)
        std = self.xi(features).clamp(-20, 2)
        return mean, std

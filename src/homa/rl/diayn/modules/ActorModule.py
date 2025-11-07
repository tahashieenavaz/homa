import torch
from torch.distributions import Categorical


class ActorModule(torch.nn.Module):
    def __init__(
        self,
        state_dimension: int,
        action_dimension: int,
        num_skills: int,
        hidden_dimension: int,
    ):
        super().__init__()
        self.state_dimension: int = state_dimension
        self.action_dimension: int = action_dimension
        self.num_skills: int = num_skills
        self.hidden_dimension: int = hidden_dimension
        self.input_dimension = self.state_dimension + self.num_skills

        self.mu = torch.nn.Sequential(
            torch.nn.Linear(self.input_dimension, self.hidden_dimension),
            torch.nn.ReLU(),
            torch.nn.Linear(self.hidden_dimension, self.hidden_dimension),
            torch.nn.ReLU(),
            torch.nn.Linear(self.hidden_dimension, self.action_dimension),
        )

    def forward(self, state: torch.Tensor, skill: torch.Tensor) -> torch.Tensor:
        phi = torch.cat([state, skill], dim=-1)
        return self.mu(phi)

    def get_action(self, state: torch.Tensor, skill: torch.Tensor):
        logits = self.forward(state, skill)
        dist = Categorical(logits=logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action, log_prob

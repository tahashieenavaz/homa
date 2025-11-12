import torch
from torch.distributions import Normal
from .modules import ContinuousActorModule
from ...common.concerns import MovesModulesToDevice


class Actor(MovesModulesToDevice):
    def __init__(
        self,
        state_dimension: int,
        action_dimension: int,
        num_skills: int,
        hidden_dimension: int,
        lr: float,
        decay: float,
        epsilon: float,
        min_std: float,
        max_std: float,
    ):
        self.epsilon: float = epsilon
        self.network = ContinuousActorModule(
            state_dimension=state_dimension,
            action_dimension=action_dimension,
            hidden_dimension=hidden_dimension,
            num_skills=num_skills,
            min_std=min_std,
            max_std=max_std,
        )
        self.optimizer = torch.optim.AdamW(
            self.network.parameters(), lr=lr, weight_decay=decay
        )
        self.move_modules()

    def action(self, state: torch.Tensor, skill: torch.Tensor):
        mean, std = self.network(state, skill)
        std = std.exp()
        distribution = Normal(mean, std)
        raw_action = distribution.rsample()
        action = torch.tanh(raw_action)
        corrected_probabilities = torch.log(1.0 - action.pow(2) + self.epsilon)
        probabilities = distribution.log_prob(raw_action) - corrected_probabilities
        probabilities = probabilities.sum(dim=-1, keepdim=True)
        return action, probabilities

    def train(self, advantages: torch.Tensor, probabilities: torch.Tensor) -> float:
        self.optimizer.zero_grad()
        loss = self.loss(advantages=advantages, probabilities=probabilities)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def loss(
        self, advantages: torch.Tensor, probabilities: torch.Tensor
    ) -> torch.Tensor:
        return -(probabilities * advantages.detach()).mean()

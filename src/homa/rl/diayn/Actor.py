import torch
from torch.distributions import Normal
from .modules import ContinuousActorModule
from ...core.concerns import MovesNetworkToDevice


class Actor(MovesNetworkToDevice):
    def __init__(
        self,
        state_dimension: int,
        action_dimension: int,
        num_skills: int,
        hidden_dimension: int,
        lr: float,
        decay: float,
        epsilon: float,
    ):
        self.network = ContinuousActorModule(
            state_dimension=state_dimension,
            action_dimension=action_dimension,
            hidden_dimension=hidden_dimension,
            num_skills=num_skills,
            epsilon=epsilon,
        )
        self.optimizer = torch.optim.AdamW(
            self.network.parameters(), lr=lr, weight_decay=decay
        )
        self.criterion = False

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

    def train(self):
        pass

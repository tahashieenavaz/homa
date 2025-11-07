import torch
import numpy
from .modules import DiscriminatorModule
from ...core.concerns import MovesNetworkToDevice


class Discriminator(MovesNetworkToDevice):
    def __init__(
        self,
        decay: float,
        lr: float,
        state_dimension: int,
        hidden_dimension: int,
        num_skills: int,
    ):
        self.network = DiscriminatorModule(
            state_dimension=state_dimension,
            hidden_dimension=hidden_dimension,
            num_skills=num_skills,
        )
        self.optimizer = torch.optim.AdamW(
            self.network.parameters(), lr=lr, weight_decay=decay
        )
        self.criterion = torch.nn.CrossEntropyLoss()

    def loss(self, states: torch.Tensor, skills_indices: torch.Tensor):
        logits = self.network(states)
        return self.criterion(logits, skills_indices)

    @torch.no_grad()
    def reward(self, state: torch.Tensor, skill_z_index: int):
        logits = self.network(state)
        log_probabilities = torch.nn.functional.log_softmax(logits, dim=-1)
        entropy_log = numpy.log(1.0 / self.num_skills)
        if skill_z_index.dim() == 1:
            skill_z_index = skill_z_index.unsqueeze(-1)
        reward = log_probabilities.gather(1, skill_z_index) - entropy_log
        return reward.squeeze(-1)

    def train(self, states: torch.Tensor, skills_indices: torch.Tensor):
        self.optimizer.zero_grad()
        loss = self.loss(states=states, skills_indices=skills_indices)
        loss.backward()
        self.optimizer.step()

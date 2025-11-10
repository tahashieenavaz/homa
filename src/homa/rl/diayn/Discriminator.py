import torch
import numpy
from .modules import DiscriminatorModule
from ...core.concerns import MovesModulesToDevice


class Discriminator(MovesModulesToDevice):
    def __init__(
        self,
        state_dimension: int,
        hidden_dimension: int,
        num_skills: int,
        decay: float,
        lr: float,
    ):
        self.num_skills: int = num_skills

        self.network = DiscriminatorModule(
            state_dimension=state_dimension,
            hidden_dimension=hidden_dimension,
            num_skills=num_skills,
        )
        self.optimizer = torch.optim.AdamW(
            self.network.parameters(), lr=lr, weight_decay=decay
        )
        self.criterion = torch.nn.CrossEntropyLoss()
        self.move_modules()

    def loss(self, states: torch.Tensor, skills_indices: torch.Tensor):
        logits = self.network(states)
        return self.criterion(logits, skills_indices)

    @torch.no_grad()
    def reward(self, state: torch.Tensor, skill_index: torch.Tensor):
        logits = self.network(state)
        probabilities = torch.nn.functional.log_softmax(logits, dim=-1)
        entropy = numpy.log(1.0 / self.num_skills)
        if skill_index.dim() == 1:
            skill_index = skill_index.unsqueeze(-1)
        reward = probabilities.gather(1, skill_index.long()) - entropy
        return reward.squeeze(-1)

    def train(self, states: torch.Tensor, skills_indices: torch.Tensor):
        self.optimizer.zero_grad()
        loss = self.loss(states=states, skills_indices=skills_indices)
        loss.backward()
        self.optimizer.step()

import torch
import numpy


class Discriminator(torch.nn.Module):
    def __init__(
        self, state_dimension: int, hidden_dimension: int = 512, num_skills: int = 10
    ):
        super().__init__()
        self.state_dimension: int = state_dimension
        self.hidden_dimension: int = hidden_dimension
        self.num_skills: int = num_skills

        self.mu = torch.nn.Sequential(
            torch.nn.Linear(self.state_dimension, self.hidden_dimension),
            torch.nn.ReLU(),
            torch.nn.Linear(self.hidden_dimension, self.hidden_dimension),
            torch.nn.ReLU(),
            torch.nn.Linear(self.hidden_dimension, self.num_skills),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.mu(state)

    @torch.no_grad()
    def get_reward(self, state: torch.Tensor, skill_z_index: int):
        logits = self.forward(state)
        log_probabilities = torch.nn.functional.log_softmax(logits, dim=-1)
        entropy_log = numpy.log(1.0 / self.num_skills)
        if skill_z_index.dim() == 1:
            skill_z_index = skill_z_index.unsqueeze(-1)
        intrinsic_reward = log_probabilities.gather(1, skill_z_index) - entropy_log
        return intrinsic_reward.squeeze(-1)

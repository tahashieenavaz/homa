import torch


class CriticModule(torch.nn.Module):
    def __init__(
        self,
        state_dimension: int,
        num_skills: int,
        hidden_dimension: int,
    ):
        super().__init__()
        self.state_dimension: int = state_dimension
        self.num_skills: int = num_skills
        self.hidden_dimension: int = hidden_dimension
        self.input_dimension = self.state_dimension + self.num_skills

        self.mu = torch.nn.Sequential(
            torch.nn.Linear(self.input_dimension, self.hidden_dimension),
            torch.nn.ReLU(),
            torch.nn.Linear(self.hidden_dimension, self.hidden_dimension),
            torch.nn.ReLU(),
            torch.nn.Linear(self.hidden_dimension, 1),
        )

    def forward(self, state: torch.Tensor, skill: torch.Tensor) -> torch.Tensor:
        phi = torch.cat([state, skill], dim=-1)
        return self.mu(phi).squeeze(-1)

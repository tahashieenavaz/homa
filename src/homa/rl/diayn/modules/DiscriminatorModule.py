import torch
import numpy


class DiscriminatorModule(torch.nn.Module):
    def __init__(self, state_dimension: int, hidden_dimension: int, num_skills: int):
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

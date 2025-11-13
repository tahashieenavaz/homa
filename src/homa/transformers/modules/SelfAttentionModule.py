import torch


class SelfAttentionModule(torch.nn.Module):
    def __init__(self, dimension: int):
        super().__init__()
        self.Q = torch.nn.Linear(dimension, dimension)
        self.K = torch.nn.Linear(dimension, dimension)
        self.V = torch.nn.Linear(dimension, dimension)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        Q = self.Q(x)
        K = self.K(x)
        V = self.V(x)

        scores = Q @ K.transpose(0, 1)
        scores = scores / (x.size(-1) ** 0.5)
        scores = torch.nn.functional.softmax(scores, dim=-1)

        return scores @ V

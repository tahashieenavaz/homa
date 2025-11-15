import torch
from .Loss import Loss


class LogitNormLoss(Loss):
    def __init__(self, epsilon: float = 1e-7):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        norms = torch.norm(logits, p=2, dim=-1, keepdim=True)
        norms = norms + self.epsilon
        normalized_logits = torch.div(logits, norms)
        return torch.nn.functional.cross_entropy(normalized_logits, target)

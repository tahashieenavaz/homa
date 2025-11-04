import torch
from .Loss import Loss


class LogitNormLoss(Loss):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, logits, target):
        norms = torch.norm(logits, p=2, dim=-1, keepdim=True) + 1e-7
        normalized_logits = torch.div(logits, norms)
        return torch.nn.functional.cross_entropy(normalized_logits, target)

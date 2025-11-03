import torch
from .ReportsLogits import ReportsLogits


class PredictsProbabilities(ReportsLogits):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.logits(x)
        return torch.nn.functional.softmax(logits, dim=1)

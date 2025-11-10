import torch
from .ReportsLogits import ReportsLogits


class PredictsProbabilities(ReportsLogits):
    def __init__(self):
        super().__init__()

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.logits(x)
        return torch.nn.functional.softmax(logits, dim=1)

    @torch.no_grad()
    def predict_(self, x: torch.Tensor) -> torch.Tensor:
        return self.predict(x)

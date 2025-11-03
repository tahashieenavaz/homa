import torch
from .concerns import (
    ReportsSize,
    RecordsStateDictionaries,
    ReportsClassificationMetrics,
)
from ..models import Model


class Ensemble(ReportsSize, ReportsClassificationMetrics, RecordsStateDictionaries):
    def __init__(self, variant: Model):
        super().__init__()
        self.model = variant().model

    def logits(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        logits = torch.zeros((batch_size, self.num_classes))
        for model in self.models:
            logits += model(x)
        return logits

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.logits(x)
        return torch.nn.functional.softmax(logits, dim=1)

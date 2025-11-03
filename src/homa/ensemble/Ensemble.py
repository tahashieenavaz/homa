import torch
from .concerns import ReportsSize, RecordsStateDictionaries


class Ensemble(ReportsSize, ReportsClassificationMetrics, RecordsStateDictionaries):
    def __init__(self):
        super().__init__()
        self.model = None

    def logits(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        logits = torch.zeros((batch_size, self.num_classes))
        for model in self.models:
            logits += model(x)
        return logits

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.logits(x)
        return torch.nn.functional.softmax(logits, dim=1)

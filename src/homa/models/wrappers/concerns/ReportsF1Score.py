import torch
from sklearn.metrics import f1_score as f1


class ReportsF1Score:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def f1(self, x: torch.Tensor, y: torch.Tensor):
        self.model.eval()
        predictions = self.model(x)
        return f1(y, predictions)

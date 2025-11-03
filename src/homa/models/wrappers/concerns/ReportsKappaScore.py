import torch
from sklearn.metrics import cohen_kappa_score as kappa


class ReportsKappaScore:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def kappa(self, x: torch.Tensor, y: torch.Tensor):
        self.model.eval()
        predictions = self.model(x)
        return kappa(y, predictions)

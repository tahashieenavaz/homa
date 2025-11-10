from sklearn.metrics import accuracy_score as accuracy
from torch.utils.data import DataLoader


class ReportsEnsembleAccuracy:
    def __init__(self):
        super().__init__()

    def accuracy(self, dataloader: DataLoader) -> float:
        predictions, labels = self.metric_necessities(dataloader)
        return accuracy(labels, predictions)

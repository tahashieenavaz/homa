from sklearn.metrics import accuracy_score as accuracy


class ReportsEnsembleAccuracy:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def accuracy(self) -> float:
        predictions, labels = self.metric_necessities()
        return accuracy(labels, predictions)

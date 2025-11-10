from sklearn.metrics import f1_score as f1


class ReportsEnsembleF1:
    def __init__(self):
        super().__init__()

    def f1(self) -> float:
        predictions, labels = self.metric_necessities()
        return f1(labels, predictions, average="weighted")

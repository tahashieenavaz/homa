from sklearn.metrics import f1_score as f1


class ReportsEnsembleF1:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def f1(self) -> float:
        labels = self.get_labels()
        predictions = self.get_predictions()
        return f1(labels, predictions, average="weighted")

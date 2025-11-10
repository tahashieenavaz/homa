from sklearn.metrics import cohen_kappa_score as kappa


class ReportsEnsembleKappa:
    def __init__(self):
        super().__init__()

    def accuracy(self) -> float:
        predictions, labels = self.metric_necessities()
        return kappa(labels, predictions)

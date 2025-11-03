from .ReportsEnsembleF1 import ReportsEnsembleF1
from .ReportsEnsembleAccuracy import ReportsEnsembleAccuracy
from .ReportsEnsembleKappa import ReportsEnsembleKappa


class ReportsClassificationMetrics(
    ReportsEnsembleF1, ReportsEnsembleAccuracy, ReportsEnsembleKappa
):
    pass

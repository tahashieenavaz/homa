from .ReportsEnsembleF1 import ReportsEnsembleF1
from .ReportsEnsembleAccuracy import ReportsEnsembleAccuracy
from .ReportsEnsembleKappa import ReportsEnsembleKappa
from .CalculatesMetricNecessities import CalculatesMetricNecessities


class ReportsClassificationMetrics(
    CalculatesMetricNecessities,
    ReportsEnsembleAccuracy,
    ReportsEnsembleF1,
    ReportsEnsembleKappa,
):
    def __init__(self):
        super().__init__()

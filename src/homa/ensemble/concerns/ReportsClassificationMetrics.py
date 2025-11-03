from .ReportsEnsembleF1 import ReportsEnsembleF1
from .ReportsEnsembleAccuracy import ReportsEnsembleAccuracy
from .ReportsEnsembleKappa import ReportsEnsembleKappa
from .CalculatesMetricNecessities import CalculatesMetricNecessities


class ReportsClassificationMetrics(
    CalculatesMetricNecessities,
    ReportsEnsembleF1,
    ReportsEnsembleAccuracy,
    ReportsEnsembleKappa,
):
    pass

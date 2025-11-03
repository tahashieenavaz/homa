from .ReportsEnsembleF1 import ReportsEnsembleF1
from .ReportsEnsembleAccuracy import ReportsEnsembleAccuracy
from .ReportsEnsembleKappa import ReportsEnsembleKappa
from .CalculatesEnsembleLabels import CalculatesEnsembleLabels
from .CalculatesEnsemblePredictions import CalculatesEnsemblePredictions


class ReportsClassificationMetrics(
    CalculatesEnsembleLabels,
    CalculatesEnsemblePredictions,
    ReportsEnsembleF1,
    ReportsEnsembleAccuracy,
    ReportsEnsembleKappa,
):
    pass

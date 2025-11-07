from .concerns import (
    ReportsEnsembleSize,
    StoresModels,
    ReportsClassificationMetrics,
    PredictsProbabilities,
)


class Ensemble(
    ReportsEnsembleSize,
    ReportsClassificationMetrics,
    PredictsProbabilities,
    StoresModels,
):
    def __init__(self):
        super().__init__()

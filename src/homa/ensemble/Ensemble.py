from .concerns import (
    ReportsEnsembleSize,
    StoresModels,
    ReportsClassificationMetrics,
    PredictsProbabilities,
    SavesEnsembleModels,
)


class Ensemble(
    ReportsEnsembleSize,
    ReportsClassificationMetrics,
    PredictsProbabilities,
    StoresModels,
    SavesEnsembleModels,
):
    def __init__(self):
        super().__init__()

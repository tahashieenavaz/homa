from .concerns import (
    ReportsSize,
    StoresModels,
    ReportsClassificationMetrics,
    PredictsProbabilities,
)


class Ensemble(
    ReportsSize,
    ReportsClassificationMetrics,
    PredictsProbabilities,
    StoresModels,
):
    def __init__(self):
        super().__init__()

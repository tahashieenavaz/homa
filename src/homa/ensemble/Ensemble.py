from .concerns import (
    ReportsSize,
    RecordsStateDictionaries,
    ReportsClassificationMetrics,
    HasNetwork,
    PredictsProbabilities,
)


class Ensemble(
    ReportsSize,
    ReportsClassificationMetrics,
    RecordsStateDictionaries,
    PredictsProbabilities,
    HasNetwork,
):
    def __init__(self):
        super().__init__()

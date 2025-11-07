import numpy
from .Buffer import Buffer
from .concerns import HasRecordAlternatives


class DiversityIsAllYouNeedBuffer(Buffer, HasRecordAlternatives):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def record(
        self,
        state: numpy.ndarray,
        action: int,
        reward: float,
        next_state: numpy.ndarray,
        termination: bool,
        log_probability: numpy.ndarray,
    ):
        self.collection.append(
            (state, action, reward, next_state, termination, log_probability)
        )

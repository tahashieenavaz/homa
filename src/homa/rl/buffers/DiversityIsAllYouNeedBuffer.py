import numpy
from .Buffer import Buffer
from .concerns import HasRecordAlternatives


class DiversityIsAllYouNeedBuffer(Buffer, HasRecordAlternatives):
    def record(
        self,
        state: numpy.ndarray,
        action: int,
        reward: float,
        next_state: numpy.ndarray,
        termination: bool,
        log_probability: numpy.ndarray,
    ):
        pass

from .concerns import ResetsCollection, SamplesItems


class Buffer(ResetsCollection, SamplesItems):
    def __init__(self, capacity: int):
        self.capacity: int = capacity
        self.reset()

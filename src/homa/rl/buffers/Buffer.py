from .concerns import ResetsCollection, HasRecordAlternatives


class Buffer(ResetsCollection, HasRecordAlternatives):
    def __init__(self, capacity: int):
        self.capacity: int = capacity
        self.reset()

    @property
    def size(self):
        return len(self.collection)

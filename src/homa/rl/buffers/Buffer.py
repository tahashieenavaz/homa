from collections import deque
from typing import Type
from .concerns import ResetsCollection, HasRecordAlternatives


class Buffer(ResetsCollection, HasRecordAlternatives):
    def __init__(self, capacity: int):
        self.capacity: int = capacity
        self.collection: Type[deque] = deque(maxlen=self.capacity)

    @property
    def size(self):
        return len(self.collection)

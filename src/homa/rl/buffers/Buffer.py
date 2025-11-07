from collections import deque


class Buffer:
    def __init__(self, capacity: int = 1000000):
        self.capacity: int = capacity
        self.collection = deque(maxlen=capacity)

from collections import deque


class ResetsCollection:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def reset(self):
        self.collection = deque(maxlen=self.capacity)

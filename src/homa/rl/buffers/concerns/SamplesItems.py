import random


class SamplesItems:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def sample(self, k: int):
        return random.sample(self.collection, k=k)

from typing import List
from collections import OrderedDict
from ...models import Model


class RecordsStateDictionaries:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.state_dicts: List[OrderedDict] = []

    def record(self, model: Model):
        self.state_dicts.append(model.network.state_dict())

    def push(self, *args, **kwargs):
        self.record(*args, **kwargs)

    def add(self, *args, **kwargs):
        self.record(*args, **kwargs)

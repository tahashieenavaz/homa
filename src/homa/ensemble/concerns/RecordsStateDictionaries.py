from typing import List
from collections import OrderedDict
from copy import deepcopy
from ...models.wrappers import ModelWrapper


class RecordsStateDictionaries:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.state_dicts: List[OrderedDict] = []

    def record(self, wrapper: ModelWrapper):
        if self.model is None:
            self.model = deepcopy(wrapper.model)
        self.state_dicts.append(wrapper.model.state_dict())

    def push(self, *args, **kwargs):
        self.record(*args, **kwargs)

    def add(self, *args, **kwargs):
        self.record(*args, **kwargs)

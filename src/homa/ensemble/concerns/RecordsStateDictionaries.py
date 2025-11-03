from typing import List
from copy import deepcopy
from collections import OrderedDict
from .HasStateDicts import HasStateDicts
from ...models import Model


class RecordsStateDictionaries(HasStateDicts):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def record(self, model: Model):
        if self.network is None:
            self.network = deepcopy(model.network)

        self.state_dicts.append(model.network.state_dict())

    def push(self, *args, **kwargs):
        self.record(*args, **kwargs)

    def append(self, *args, **kwargs):
        self.record(*args, **kwargs)

    def add(self, *args, **kwargs):
        self.record(*args, **kwargs)

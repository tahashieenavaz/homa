from copy import deepcopy
from collections import OrderedDict
from .HasStateDicts import HasStateDicts
from ...vision import Model


class RecordsStateDictionaries(HasStateDicts):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def record(self, model: Model | OrderedDict):
        state_dict_to_add: OrderedDict | None = None

        if isinstance(model, Model):
            if self.network is None:
                self.network = deepcopy(model.network)
            state_dict_to_add = model.network.state_dict()
        elif isinstance(model, OrderedDict):
            state_dict_to_add = model
        elif isinstance(model, torch.nn.Module):
            state_dict_to_add = model.state_dict()
        else:
            raise TypeError("Invalid input for ensemble record.")

        self.state_dicts.append(state_dict_to_add)

    def push(self, *args, **kwargs):
        self.record(*args, **kwargs)

    def append(self, *args, **kwargs):
        self.record(*args, **kwargs)

    def add(self, *args, **kwargs):
        self.record(*args, **kwargs)

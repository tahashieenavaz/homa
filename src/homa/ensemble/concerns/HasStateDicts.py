from typing import List
from collections import OrderedDict


class HasStateDicts:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.state_dicts: List[OrderedDict] = []

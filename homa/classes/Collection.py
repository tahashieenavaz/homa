from typing import List
from typing import Tuple


class Collection:
    def __init__(self, items: List[any] | Tuple[any]):
        self.value = items

    def map(self, callback: callable):
        return list(
            map(callback, self.value)
        )

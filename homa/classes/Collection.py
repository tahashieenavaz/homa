from typing import List


class Collection:
    def __init__(self, items: List[any]):
        self.value = items

    def map(self, callback: callable):
        return list(
            map(callback, self.value)
        )

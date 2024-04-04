import sys
from typing import List
from .classes.Collection import Collection
from .classes.Logger import Logger


def is_colab() -> bool:
    return 'google.colab' in sys.modules


def collection(items: List[any]):
    return Collection(items)


def danger(message: str) -> None:
    Logger.danger(message)


def create_kernel(value: int | tuple) -> tuple:
    if isinstance(value, tuple):
        x, y = value

        x = x - 1 if x % 2 == 0 else x
        y = y - 1 if y % 2 == 0 else y

        return (x, y)

    value = value - 1 if value % 2 == 0 else value

    return (value, value)

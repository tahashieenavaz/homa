from typing import Tuple


def create_kernel(value: int | Tuple[int, int]) -> Tuple[int, int]:
    if isinstance(value, tuple):
        x, y = value

        x = x - 1 if x % 2 == 0 else x
        y = y - 1 if y % 2 == 0 else y

        return (x, y)

    value = value - 1 if value % 2 == 0 else value

    return (value, value)

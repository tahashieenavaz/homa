from typing import Tuple
from .helpers.alias import repo


def rgb(key: str) -> Tuple[int, int, int]:
    image = repo(key)

    if image.shape[2] == 3:
        # found three channels in stored image
        # meaning that it has been loaded as a color image

        b = image[:, :, 0]
        g = image[:, :, 0]
        r = image[:, :, 0]

        return (r, g, b)

    if image.shape[2] == 1:
        gray = image[:, :, 0]
        return (gray, gray, gray)

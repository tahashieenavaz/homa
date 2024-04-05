from typing import Tuple
from .helpers.alias import repo
import cv2


def hsv(key: str):
    image = repo(key)
    cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    h = image[:, :, 0]
    s = image[:, :, 1]
    v = image[:, :, 2]

    return (h, s, v)


def bgr(key: str) -> Tuple[int, int, int]:
    return rgb(key, bgr_flag=True)


def rgb(key: str, bgr_flag: bool = False) -> Tuple[int, int, int]:
    image = repo(key)

    if image.shape[2] == 3:
        # found three channels in stored image
        # meaning that it has been loaded as a color image

        b = image[:, :, 0]
        g = image[:, :, 0]
        r = image[:, :, 0]

        if bgr_flag:
            return (b, g, r)

        return (r, g, b)

    if image.shape[2] == 1:
        gray = image[:, :, 0]
        return (gray, gray, gray)

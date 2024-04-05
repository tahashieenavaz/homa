import cv2
from .main import setting
from .main import refresh
from .helpers.alias import repo
from typing import Tuple
import numpy


def stroke(value: int = 1):
    setting("thickness", value)


def fill(*args):
    setting("thickness", -1)

    if len(args) == 3:
        color(*args)


def randomColor() -> Tuple[int, int, int]:
    r = numpy.random.randint(0, 255)
    g = numpy.random.randint(0, 255)
    b = numpy.random.randint(0, 255)

    return (b, g, r)


def color(*args):
    # no colors provided, choose a random color
    if len(args) == 0:
        args = randomColor()

    setting("color", args)


def circle(key: str, x: int, y: int, radius: int = 1):
    cv2.circle(
        repo(key),
        (x, y),
        radius,
        setting("color"),
        setting("thickness")
    )
    refresh(key)


def rect(key: str, x: int, y: int, width: int, height: int):
    cv2.rectangle(
        repo(key),
        (x - width // 2, y - height // 2), (x + width // 2, y + height // 2),
        thickness=setting("thickness"),
        color=setting("color")
    )
    refresh(key)

import cv2
from .main import setting
from .main import refresh
from .helpers.alias import repo
from .classes.Repository import Repository
from typing import Tuple


def stroke(value: int = 1):
    setting("thickness", value)


def fill():
    setting("thickness", -1)


def color(*args):
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

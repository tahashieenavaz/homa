import cv2
from .main import refresh
from .helpers.alias import repo
from .classes.Repository import Repository
from .helpers.alias import setting


def thickness(value: int = 1):
    setting("thickness", value)


def circle(key: str, x: int = 0, y: int = 0, radius: int = 1, color=(0, 0, 255)):
    cv2.circle(repo(key), (x, y), radius, color, setting("thickness"))
    refresh(key)

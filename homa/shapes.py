import cv2
from .main import refresh
from .helpers.alias import repo
from .classes.Repository import Repository


def circle(key: str, x: int = 0, y: int = 0, r: int = 1, color=(0, 0, 255), thickness: int = 1):
    cv2.circle(repo(key), (x, y), r, color, thickness)
    refresh(key)

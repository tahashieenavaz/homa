from typing import List
from .classes.Repository import Repository
from .helpers.kernel import create_kernel
import cv2


def blur(key: str, kernel: int | List[int] = (7, 7), new_key: str | None = None) -> None:
    if new_key is None:
        new_key = key

    Repository.images[new_key] = cv2.blur(
        Repository.images[key],
        create_kernel(kernel)
    )


def median(key: str, kernel: int | List[int] = (7, 7), new_key: str | None = None) -> None:
    if new_key is None:
        new_key = key

    Repository.images[new_key] = cv2.medianBlur(
        Repository.images[key],
        create_kernel(kernel)
    )


def sigma(x: float = 0, y: float = 0) -> None:
    Repository.sigmaX = x
    Repository.sigmaY = y


def gaussian(key: str, kernel: None | List[int] = None, new_key: str | None = None) -> None:
    if new_key is None:
        new_key = key

    Repository.images[new_key] = cv2.GaussianBlur(
        Repository.images[key],
        create_kernel(kernel),
        sigmaX=Repository.sigmaX,
        sigmaY=Repository.sigmaY
    )

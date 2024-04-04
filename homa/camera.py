import cv2
from .classes.Repository import Repository
from typing import Iterator


def camera(delay: int = 10, exit_on: str = "q") -> Iterator[int]:
    pressed_key = None
    capture = cv2.VideoCapture(0)
    frame_number = 0
    while pressed_key != ord(exit_on):
        _, frame = capture.read()
        Repository.images["camera"] = frame

        frame_number += 1
        yield frame_number

        pressed_key = cv2.waitKey(delay)

    capture.release()

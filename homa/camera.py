import cv2
from typing import Iterator
from .classes.Repository import Repository
from .classes.Window import Window


def camera(delay: int = 10, exit_on: str = "q") -> Iterator[int]:
    cameraWindow = Window(640, 480)

    pressed_key = None
    capture = cv2.VideoCapture(0)
    frame_number = 0
    while pressed_key != ord(exit_on):
        _, frame = capture.read()
        cameraWindow.update(frame)

        frame_number += 1
        yield frame_number

        pressed_key = cv2.waitKey(delay)

    capture.release()

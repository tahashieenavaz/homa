import cv2
from typing import Iterator
from .classes.Window import Window


def camera(delay: int = 10, exitOn: str = "q", onClick: callable = None) -> Iterator[int]:
    cameraWindow = Window(640, 480)

    if onClick is not None:
        cameraWindow.click(onClick)

    pressed_key = None
    capture = cv2.VideoCapture(0)
    frame_number = 0
    while pressed_key != ord(exitOn):
        _, frame = capture.read()
        cameraWindow.update(frame)
        frame_number += 1
        yield cameraWindow
        pressed_key = cv2.waitKey(delay)

    capture.release()

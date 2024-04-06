import cv2
from typing import Iterator
from .classes.Window import Window


def camera(
    delay: int = 10,
    exitOn: str = "q",
    onClick: callable = None,
    onMove: callable = None,
) -> Iterator[int]:

    cameraWindow = Window(640, 480)

    if onClick is not None:
        cameraWindow.click(onClick)

    if onMove is not None:
        cameraWindow.move(onMove)

    pressedKey = None
    capture = cv2.VideoCapture(0)
    frameNumber = 0
    while pressedKey != ord(exitOn):
        _, frame = capture.read()
        cameraWindow.update(frame)
        frameNumber += 1
        yield (cameraWindow, frameNumber)
        pressedKey = cv2.waitKey(delay)

    capture.release()

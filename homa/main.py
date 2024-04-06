import cv2
import numpy
from .classes.Window import Window
from .classes.Repository import Repository


def destroy(key: str | None = None) -> None:
    if key is not None:
        cv2.destroyWindow(key)
        return

    cv2.destroyAllWindows()


def show(*windows, **settings):
    for window in windows:
        if isinstance(window, numpy.ndarray):
            window = Window(image=window)
        window.show()

    if "wait" in settings and settings["wait"]:
        cv2.waitKey()


def showWait(*args, **kwargs):
    kwargs = {
        **kwargs,
        "wait": True
    }
    show(*args, **kwargs)

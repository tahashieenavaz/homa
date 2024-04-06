from .classes.Window import Window
from .classes.WindowRepository import WindowRepository
from .helpers.alias import collection
import numpy


def stack(*windows, **settings) -> Window:
    defaultSettings = {
        "axis": 1,
    }
    settings = {
        **defaultSettings,
        **settings,
    }

    def mapHandler(x):
        if isinstance(x, Window):
            return x.getImage()
        elif isinstance(x, numpy.ndarray):
            return x

    rawImages = list(map(mapHandler, windows))
    return numpy.concatenate(rawImages, axis=settings["axis"])


def vstack(*windows):
    return stack(*windows, axis=1)


def hstack(*windows):
    return stack(*windows, axis=0)

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

    rawImages = list(map(lambda x: x.getImage(), windows))
    combinedImage = numpy.concatenate(rawImages, axis=settings["axis"])
    windowKey = hash(f"{str(windows)}{settings['axis']}")
    window = WindowRepository.windowGetOrCreate(windowKey)
    window.update(combinedImage)
    return window


def vstack(*windows):
    return stack(*windows, axis=1)


def hstack(*windows):
    return stack(*windows, axis=0)

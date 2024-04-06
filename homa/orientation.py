from .classes.Window import Window
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

    images = collection(windows).map(lambda x: x.getImage())
    combined = numpy.concatenate(images, axis=settings["axis"])
    window = Repository.windowGetOrCreate()
    return Window(
        image=combined,
    )


def vstack(*windows):
    return stack(*windows, axis=1)


def vstack(*windows):
    return stack(*windows, axis=0)

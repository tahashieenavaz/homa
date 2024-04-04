from typing import List
from .classes.Repository import Repository
from .helpers import collection
import numpy


def stack(*keys, **settings):
    default_settings = {
        "axis": 1,
        "new_key": None
    }

    settings = {
        **default_settings,
        **settings
    }

    stacked_image = numpy.concatenate(
        collection(keys).map(lambda key: Repository.images[key]),
        axis=settings["axis"]
    )

    if settings["new_key"] is not None:
        Repository.images[settings["new_key"]] = stacked_image

    return stacked_image


def vstack(*keys, **settings):
    settings["axis"] = 1
    return stack(*keys, **settings)


def hstack(*keys, **settings):
    settings["axis"] = 0
    return stack(*keys, **settings)

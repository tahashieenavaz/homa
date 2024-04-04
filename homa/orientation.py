from .classes.Repository import Repository
from .helpers.alias import collection
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

    if all(isinstance(item, str) for item in keys):
        keys = collection(keys).map(lambda key: Repository.images[key])

    stacked_image = numpy.concatenate(
        keys,
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

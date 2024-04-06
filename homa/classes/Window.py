from ..main import repo
from ..main import black
import string
import numpy


class Window:
    def __init__(self, width: int = 300, height: int = 300, image=None, channels: int = 3, dtype="uint8") -> None:
        self.width = width
        self.height = height

        if image is None:
            image = numpy.zeros([height, width, channels], dtype=dtype)

        self.image = image
        self.events = {}

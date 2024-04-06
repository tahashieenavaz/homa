import numpy


class Window:
    def __init__(self, width: int = 300, height: int = 300, image=None, title: str|None = None, channels: int = 3, dtype="uint8") -> None:
        if title is None:

        if image is None:
            image = numpy.zeros([height, width, channels], dtype=dtype)

        self.width = width
        self.height = height
        self.image = image
        self.events = {}

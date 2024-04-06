import numpy
import cv2
from ..helpers.string import randomLowercaseString


class Window:
    def __init__(self, width: int = 300, height: int = 300, image=None, title: str | None = None, channels: int = 3, dtype="uint8") -> None:
        if title is None:
            title = randomLowercaseString(10)

        if image is None:
            image = numpy.zeros([height, width, channels], dtype=dtype)

        self.__title = title
        self.__width = width
        self.__height = height
        self.__image = image
        self.__events = {}

        cv2.namedWindow(self.__title)

    def title(self, newTitle: str):
        self.__title = newTitle
        return self

    def show(self):
        cv2.setMouseCallback(self.__title, createMouseEvent(self.__events))
        cv2.imshow(self.__title, self.__image)

    def onClick(self, handler: callable):
        pass

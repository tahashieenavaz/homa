import numpy
import cv2
from ..classes.Repository import Repository
from ..helpers.string import randomLowercaseString
from ..events import createMouseCallback
from typing_extensions import Self


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

    def show(self):
        cv2.namedWindow(self.__title)
        cv2.setMouseCallback(self.__title, createMouseCallback(self.__events))
        Repository.imshow(self.__title, self.__image)

    def title(self, newTitle: str) -> Self:
        self.__title = newTitle
        return self

    def update(self, newImage) -> Self:
        self.__image = newImage
        return self

    def click(self, handler: callable) -> Self:
        self.__events["click"] = handler
        return self

    def move(self, handler: callable) -> Self:
        self.__events["mousemove"] = handler
        return self

    def __getattr__(self, key):
        if key == "shape":
            return (self.__width, self.__height)

        return None

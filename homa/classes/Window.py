import numpy
import cv2

from ..helpers.kernel import createKernel
from ..helpers.string import randomLowercaseString

from ..classes.Repository import Repository

from ..events import createMouseCallback

from typing_extensions import Self
from typing import List

from ..shapes import color
from ..shapes import stroke

from ..main import setting


class Window:
    def __init__(self, width: int = 300, height: int = 300, image=None, title: str | None = None, channels: int = 3, dtype="uint8") -> None:
        if title is None:
            title = randomLowercaseString(10)

        if image is None:
            image = numpy.zeros([height, width, channels], dtype=dtype)

        self.__title = title
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
        self.refresh()
        return self

    def click(self, handler: callable) -> Self:
        self.__events["click"] = (handler, self)
        return self

    def move(self, handler: callable) -> Self:
        self.__events["mousemove"] = (handler, self)
        return self

    def white(self) -> Self:
        self.__image = numpy.ones([
            self.__image.shape[0],
            self.__image.shape[1],
            self.__image.shape[2]
        ]) * (2 ** 8 - 1)
        return self

    def blur(self, kernel: int | List[int] = (7, 7)) -> Self:
        self.update(cv2.blur(
            self.__image,
            createKernel(kernel),
        ))

        return self

    def gaussian(self, kernel: int | List[int] = (7, 7)) -> Self:
        self.update(cv2.GaussianBlur(
            self.__image,
            createKernel(kernel),
            setting("sigma")[0],
            setting("sigma")[1],
        ))

        return self

    def median(self, kernel: int) -> Self:
        self.update(cv2.medianBlur(
            self.__image, kernel
        ))

        return self

    def refresh(self):
        Repository.imshow(self.__title, self.__image)

    def circle(self, x: int, y: int, radius: int, circleColor: tuple | None = None, thickness: int | None = None):
        if circleColor is not None:
            color(*circleColor)

        if thickness is not None:
            stroke(thickness)

        self.update(cv2.circle(
            self.__image,
            (x, y),
            radius,
            setting("color"),
            setting("thickness")
        ))

    def getImage(self):
        return self.__image

    def getTitle(self):
        return self.__title

    def __getattr__(self, key):
        if key == "shape":
            return self.__image.shape

        return None

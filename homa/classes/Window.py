import numpy
import cv2

from ..helpers.kernel import createKernel
from ..helpers.string import randomLowercaseString
from ..helpers.environment import isNotColab
from ..helpers.alias import setting

from ..classes.Repository import Repository

from ..events import createMouseCallback

from typing_extensions import Self
from typing import Tuple

from ..classes.Circle import Circle


class Window:
    def __init__(self, width: int = 300, height: int = 300, image=None, title: str | None = None, channels: int = 3, dtype="uint8") -> None:
        if title is None:
            title = randomLowercaseString(10)

        if image is None:
            image = numpy.zeros([height, width, channels], dtype=dtype)

        self.__title = title
        self.__image = image
        self.__events = {}

        self.__circles = []

    def title(self, newTitle: str) -> Self:
        self.__title = newTitle
        return self

    def show(self):
        if isNotColab():
            cv2.namedWindow(self.__title)
            if self.__events != {}:
                cv2.setMouseCallback(
                    self.__title,
                    createMouseCallback(self.__events)
                )

        self.drawCircles()
        Repository.imshow(self.__title, self.__image)

    def drawCircles(self):
        for circle in self.__circles:
            cv2.circle(
                self.__image,
                (circle.x, circle.y),
                circle.radius,
                circle.color,
                circle.stroke
            )

    def update(self, newImage):
        self.__image = newImage
        self.refresh()

    def click(self, handler: callable):
        self.__events["click"] = (handler, self)

    def move(self, handler: callable):
        self.__events["mousemove"] = (handler, self)

    def white(self):
        self.__image = numpy.ones([
            self.__image.shape[0],
            self.__image.shape[1],
            self.__image.shape[2]
        ]) * (2 ** 8 - 1)

    def blur(self, kernelX: int, kernelY: int | None = None, inplace: bool = True):
        kernel = (kernelX, kernelY)
        if kernelY is None:
            kernel = (kernelX, kernelX)
        blurredImage = self.__image.copy()
        blurredImage = cv2.blur(
            blurredImage,
            createKernel(kernel),
        )
        if inplace:
            self.update(blurredImage)
        return blurredImage

    def blurred(self, *args, **kwargs):
        kwargs = {
            **kwargs,
            "inplace": False
        }
        return self.blur(*args, **kwargs)

    def gaussian(self, kernelX: int, kernelY: int | None = None, inplace: bool = True):
        kernel = (kernelX, kernelY)
        if kernelY is None:
            kernel = (kernelX, kernelX)
        gaussianedImage = self.__image.copy()
        gaussianedImage = cv2.GaussianBlur(
            gaussianedImage,
            createKernel(kernel),
            setting("sigma")[0],
            gaussianedImage,
            setting("sigma")[1]
        )
        if inplace:
            self.update(gaussianedImage)
        return gaussianedImage

    def gaussianed(self, *args, **kwargs):
        kwargs = {
            **kwargs,
            "inplace": False
        }
        return self.gaussian(*args, **kwargs)

    def median(self, kernel: int, inplace: bool = True):
        kernel = kernel - 1 if kernel % 2 == 0 else kernel
        medianedImage = self.__image.copy()
        medianedImage = cv2.medianBlur(medianedImage, kernel)
        print("here")
        if inplace:
            self.update(medianedImage)
        return medianedImage

    def medianed(self, *args, **kwargs):
        kwargs = {
            **kwargs,
            "inplace": False
        }
        return self.median(*args, **kwargs)

    def refresh(self):
        Repository.imshow(self.__title, self.__image)

    def circle(self, x: int, y: int, radius: int, color: Tuple[int, int, int] | None = None, stroke: int | None = None):
        self.__circles.append(Circle(x, y, radius, color, stroke))

    def getImage(self):
        return self.__image

    def getTitle(self):
        return self.__title

    def __getattr__(self, key):
        if key == "shape":
            return self.__image.shape

        return None

from .Window import Window
from ..helpers.string import randomLowercaseString
import cv2


class Image(Window):
    def __init__(self, filename: str | Window):
        if isinstance(filename, Window):
            image = filename.getImage()
            title = f"{filename.getTitle()} Cloned"

        elif isinstance(filename, str):
            image = cv2.imread(filename)
            title = filename

        super().__init__(
            image=image,
            title=title
        )

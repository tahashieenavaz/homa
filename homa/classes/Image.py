from .Window import Window
import cv2


class Image(Window):
    def __init__(self, filename: str):
        super().__init__(
            image=cv2.imread(filename),
            title=filename
        )

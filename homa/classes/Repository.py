from ..helpers.environment import is_colab
import string
import numpy


class RepositoryWrapper:
    def __init__(self):
        self.directory = "./"
        self.images = {}

        self.settings = {
            "thickness": 2,
            "color": (0, 0, 0),
            "sigma": [0, 0]
        }

        self.events = {}

        if is_colab():
            from google.colab.patches import cv2_imshow as imshow
        else:
            from cv2 import imshow

        def final_imshow(window, image):
            if is_colab():
                imshow(image)
            else:
                imshow(window, image)

        self.imshow = final_imshow

    def addImageWithRandomKey(self, image):
        key = "".join(numpy.random.choice(
            list(string.ascii_lowercase), 10))

        self.addImage(key, image)

    def addImage(self, key, image):
        Repository.images[key] = image


Repository = RepositoryWrapper()

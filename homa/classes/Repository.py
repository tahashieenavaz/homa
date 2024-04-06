from ..helpers.environment import isColab
from ..helpers.string import randomLowercaseString


class RepositoryWrapper:
    def __init__(self):
        self.directory = "./"

        self.settings = {
            "thickness": 2,
            "color": (0, 0, 0),
            "sigma": [0, 0]
        }

        if isColab():
            from google.colab.patches import cv2_imshow as imshow
        else:
            from cv2 import imshow

        def final_imshow(window, image):
            if isColab():
                imshow(image)
            else:
                imshow(window, image)

        self.imshow = final_imshow

    def addImageWithRandomKey(self, image):
        self.addImage(randomLowercaseString(), image)

    def addImage(self, key, image):
        Repository.images[key] = image


Repository = RepositoryWrapper()

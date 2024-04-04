from ..helpers.environment import is_colab


class RepositoryWrapper:
    def __init__(self):
        self.sigmaX = 0
        self.sigmaY = 0

        self.directory = "./"
        self.images = {}
        self.windows = {}

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


Repository = RepositoryWrapper()

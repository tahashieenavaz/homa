from homa import *

import unittest


class LoadingImagesTest(unittest.TestCase):
    def setUp(self):
        pass

    def test_can_load_images_into_repository(self):
        image("italy.jpg", "italy")
        self.assertIn("italy", repo())

    def test_it_does_not_load_extra_images(self):
        image("italy.jpg", "italy")
        self.assertEqual(len(repo()), 1)

        image("italy.jpg", "italy again")
        self.assertEqual(len(repo()), 2)

    def test_it_can_load_images_as_black_and_white(self):
        image("italy.jpg", "italy", color=False)
        self.assertEqual(len(repo("italy").shape), 2)

    def test_it_can_load_images_as_colorful(self):
        image("italy.jpg", "italy", color=True)
        self.assertEqual(len(repo("italy").shape), 3)

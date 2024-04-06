import numpy
import string


def randomLowercaseString(length: int = 10):
    return randomString(length, string.ascii_lowercase)


def randomUppercaseString(length: int = 10):
    return randomString(length, string.ascii_uppercase)


def randomString(length: int = 10, letters: str = string.ascii_letters):
    return "".join(
        numpy.random.choice(list(letters), length)
    )

import sys


def isColab() -> bool:
    return 'google.colab' in sys.modules


def isNotColab() -> bool:
    return not isColab()

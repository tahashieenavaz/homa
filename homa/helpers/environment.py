import sys


def is_colab() -> bool:
    return 'google.colab' in sys.modules

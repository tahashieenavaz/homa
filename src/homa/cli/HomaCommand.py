import fire
from .namespaces import MakeNamespace, CacheNamespace


class HomaCommand:
    def __init__(self):
        self.make = MakeNamespace()
        self.cache = CacheNamespace()


def main():
    fire.Fire(HomaCommand)

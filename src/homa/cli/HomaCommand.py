import fire
from .namespaces import MakeNamespace, CacheNamespace, InitNamespace


class HomaCommand:
    def __init__(self):
        self.make = MakeNamespace()
        self.cache = CacheNamespace()
        self.init = InitNamespace()


def main():
    fire.Fire(HomaCommand)

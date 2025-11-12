import fire
from .Namespaces import MakeNamespace, CacheNamespace, MediaNamespace
from .Commands import InitCommand


class HomaCommand:
    def __init__(self):
        self.make = MakeNamespace()
        self.cache = CacheNamespace()
        self.media = MediaNamespace()

    def init(self):
        InitCommand.run()


def main():
    fire.Fire(HomaCommand)

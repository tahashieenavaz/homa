import fire
from .namespaces import MakeNamespace, CacheNamespace
from .Commands import InitCommand


class HomaCommand:
    def __init__(self):
        self.make = MakeNamespace()
        self.cache = CacheNamespace()

    def init(self):
        InitCommand.run()


def main():
    fire.Fire(HomaCommand)

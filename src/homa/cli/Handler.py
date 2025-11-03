from .CommandTable import CommandTable
from ..utils import invoke


class Handler:
    def __init__(self):
        self.command_table = invoke(CommandTable)

    @staticmethod
    def main():
        pass

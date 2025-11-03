from .Command import Command


class MakeClassCommand(Command):
    def __call__(self):
        print("making a class")

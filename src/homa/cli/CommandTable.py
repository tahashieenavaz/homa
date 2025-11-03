from .commands import RemovePyCacheCommand, MakeClassCommand


class CommandTable:
    def __call__(self):
        return {"cache:remove": RemovePyCacheCommand, "make:class": MakeClassCommand}

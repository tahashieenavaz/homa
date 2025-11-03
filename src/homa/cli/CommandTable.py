from .commands import RemovePyCacheCommand


class CommandTable:
    def __call__(self, *args, **kwargs):
        return {"cache:remove": RemovePyCacheCommand}

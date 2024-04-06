from .Window import Window


class WindowRepositoryWrapper:
    def __init__(self) -> None:
        self.windows = {}

    def windowGetOrCreate(self, key: str):
        if not key in self.windows:
            self.windows[key] = Window()

        return self.windows[key]


WindowRepository = WindowRepositoryWrapper()

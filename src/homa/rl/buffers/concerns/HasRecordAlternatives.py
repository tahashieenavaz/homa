class HasRecordAlternatives:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def add(self, *args, **kwargs) -> None:
        self.record(*args, **kwargs)

    def push(self, *args, **kwargs) -> None:
        self.record(*args, **kwargs)

    def append(self, *args, **kwargs) -> None:
        self.record(*args, **kwargs)

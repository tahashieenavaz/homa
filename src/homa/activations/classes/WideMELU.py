from .MexicanReLU import MexicanReLU


class WideMeLU(MexicanReLU):
    def __init__(self, channels: int | None = None, max_input: float = 1.0):
        self.hats = [
            (2.0, 2.0),
            (1.0, 1.0),
            (3.0, 1.0),
            (0.5, 0.5),
            (1.5, 0.5),
            (2.5, 0.5),
            (3.5, 0.5),
        ]
        super().__init__(self.hats, channels=channels, max_input=max_input)

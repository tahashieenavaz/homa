from .GaussianReLU import GaussianReLU


class GaLU(GaussianReLU):
    def __init__(
        self,
        channels: int | None = None,
        max_input: float = 1.0,
    ):
        self.hats = [(2.0, 2.0), (1.0, 1.0), (3.0, 1.0)]
        super().__init__(self.hats, channels=channels, max_input=max_input)

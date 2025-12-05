from .NReLU import NReLU


class NoisyReLU(NReLU):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

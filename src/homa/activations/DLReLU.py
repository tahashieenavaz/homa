from .BaseDLReLU import BaseDLReLU


class DLReLU(BaseDLReLU):
    def __init__(self, a: float = 0.01, init_mse: float = 1.0):
        super().__init__(a=a, init_mse=init_mse, mode="linear")

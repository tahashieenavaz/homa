from .TBSReLU import TBSReLU


class TangentBipolarSigmoidReLU(TBSReLU):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

from ...device import move


class MovesNetworkToDevice:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if not hasattr(self, "network"):
            raise RuntimeError(
                "MovesNetworkToDevice assumes the underlying class has a network property."
            )

        move(self.network)

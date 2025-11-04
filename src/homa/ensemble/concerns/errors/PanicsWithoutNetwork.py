class PanicsWithoutNetwork:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def panic_if_no_network(self):
        if self.network is None:
            raise ValueError("An empty ensemble cannot generate logits")

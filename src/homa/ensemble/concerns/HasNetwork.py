class HasNetwork:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # This property is going to be filled with the first model that is fed into the ensemble.
        self.network = None

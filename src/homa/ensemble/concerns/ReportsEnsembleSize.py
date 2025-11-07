class ReportsEnsembleSize:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def size(self):
        return len(self.weights)

    @property
    def length(self):
        return self.size

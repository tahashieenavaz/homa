class ReportsEnsembleSize:
    def __init__(self):
        super().__init__()

    @property
    def size(self):
        return len(self.weights)

    @property
    def length(self):
        return self.size

class ReportsSize:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def size(self):
        return len(self.state_dicts)

    @property
    def length(self):
        return len(self.state_dicts)

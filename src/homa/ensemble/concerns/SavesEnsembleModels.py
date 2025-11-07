class SavesEnsembleModels:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def save(self):
        self.save_factories()
        self.save_weights()

    def save_factories(self):
        pass

    def save_weights(self):
        pass

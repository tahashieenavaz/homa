class TracksTime:
    def __init__(self):
        super().__init__()
        self.t = 0

    def tick(self):
        self.t += 1

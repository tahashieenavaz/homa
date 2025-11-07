from ..device import move
from .modules import GraphAttentionModule


class GraphAttention:
    def __init__(self):
        super().__init__()
        self.network = GraphAttentionModule()
        move(self.network)

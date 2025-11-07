from .modules import GraphAttentionModule
from ..core.concerns import MovesNetworkToDevice


class GraphAttention(MovesNetworkToDevice):
    def __init__(self):
        super().__init__()
        self.network = GraphAttentionModule()

import torch
from .modules import GraphAttentionModule
from ..core.concerns import MovesNetworkToDevice


class GraphAttention(MovesNetworkToDevice):
    def __init__(self, lr: float = 0.005, decay: float = 5e-4, dropout: float = 0.6):
        super().__init__()
        self.network = GraphAttentionModule()
        self.optimizer = torch.nn.AdamW(
            self.network.parameters(), lr=lr, weight_decay=decay
        )
        self.criterion = torch.nn.CrossEntropyLoss()

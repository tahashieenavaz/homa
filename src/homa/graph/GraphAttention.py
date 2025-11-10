import torch
from .modules import GraphAttentionModule
from ..core.concerns import MovesModulesToDevice


class GraphAttention(MovesModulesToDevice):
    def __init__(
        self,
        input_dimension: int,
        output_dimension: int,
        hidden_dimension: int = 8,
        num_heads: int = 8,
        alpha: float = 0.2,
        lr: float = 0.005,
        decay: float = 0.0005,
        dropout: float = 0.6,
        concat: bool = True,
    ):
        super().__init__()

        self.network = GraphAttentionModule(
            input_dimension=input_dimension,
            output_dimension=output_dimension,
            hidden_dimension=hidden_dimension,
            num_heads=num_heads,
            dropout=dropout,
            alpha=alpha,
            concat=concat,
        )
        self.optimizer = torch.nn.AdamW(
            self.network.parameters(), lr=lr, weight_decay=decay
        )
        self.criterion = torch.nn.CrossEntropyLoss()
        self.move_modules()

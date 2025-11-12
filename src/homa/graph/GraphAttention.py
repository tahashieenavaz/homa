import torch
from .modules import GraphAttentionModule
from ..common.concerns import MovesModulesToDevice


class GraphAttention(MovesModulesToDevice):
    def __init__(
        self,
        labels: torch.Tensor,
        features: torch.Tensor,
        adjacency_matrix: torch.Tensor,
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

        self.features = features
        self.adjacency_matrix = adjacency_matrix
        self.labels = labels

        self.network = GraphAttentionModule(
            input_dimension=input_dimension,
            output_dimension=output_dimension,
            hidden_dimension=hidden_dimension,
            num_heads=num_heads,
            dropout=dropout,
            alpha=alpha,
            concat=concat,
        )
        self.move_modules()

        self.optimizer = torch.optim.AdamW(
            self.network.parameters(), lr=lr, weight_decay=decay
        )

        # this is used because model outputs log-probability distribution
        self.criterion = torch.nn.NLLLoss()

        # this will be used to share data between loss and accuracy methods
        self._predictions = None

    def train(self, idx: torch.Tensor):
        self.network.train()
        self.optimizer.zero_grad()
        loss = self.loss(idx=idx)
        loss.backward()
        self.optimizer.step()

    def loss(
        self,
        idx,
    ):
        self._predictions = self.network(self.features, self.adjacency_matrix)
        masked_predictions = self._predictions[idx]
        masked_labels = self.labels[idx]
        return self.criterion(masked_predictions, masked_labels)

    @torch.no_grad()
    def accuracy(self, mask: torch.tensor):
        self.network.eval()
        masked_predictions = self._predictions[mask].argmax(dim=1)
        masked_labels = self.labels[mask]
        return (masked_predictions == masked_labels).float().mean()

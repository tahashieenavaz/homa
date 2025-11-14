import torch
from sklearn.metrics import f1_score, cohen_kappa_score
from types import SimpleNamespace
from typing import Type, OrderedDict
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
        activation: torch.nn.Module = torch.nn.LeakyReLU,
        final_activation: torch.nn.Module = torch.nn.ELU,
        v2: bool = False,
        use_layernorm: bool = False,
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
            activation=activation,
            final_activation=final_activation,
            v2=v2,
            use_layernorm=use_layernorm,
        )
        self.move_modules()

        self.optimizer = torch.optim.AdamW(
            self.network.parameters(), lr=lr, weight_decay=decay
        )

        # this is used because model outputs log-probability distribution
        self.criterion = torch.nn.NLLLoss()

    def train(self, mask: torch.Tensor) -> None:
        self.network.train()
        self.optimizer.zero_grad()
        loss = self.loss(mask=mask)
        loss.backward()
        self.optimizer.step()

    def loss(self, mask: torch.Tensor) -> torch.Tensor:
        predictions = self.network(self.features, self.adjacency_matrix)
        masked_predictions = predictions[mask]
        masked_labels = self.labels[mask]
        return self.criterion(masked_predictions, masked_labels)

    def state_dict(self):
        return self.network.state_dict()

    def load_state_dict(self, state_dict: OrderedDict):
        self.network.load_state_dict(state_dict)

    @torch.no_grad()
    def metrics(self, mask: torch.Tensor) -> Type[SimpleNamespace]:
        self.network.eval()

        predictions = self.network(self.features, self.adjacency_matrix)
        masked_predictions = predictions[mask].argmax(dim=1).cpu()
        masked_labels = self.labels[mask].cpu()

        accuracy = (masked_predictions == masked_labels).float().mean().item()
        f1_macro = f1_score(masked_labels, masked_predictions, average="micro")
        kappa = cohen_kappa_score(masked_labels, masked_predictions)

        return SimpleNamespace(
            **{
                "accuracy": accuracy,
                "f1": f1_macro,
                "kappa": kappa,
            }
        )

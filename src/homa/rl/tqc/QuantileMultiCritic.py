import torch
from .modules import QuantileCriticModule


class QuantileMultiCritic:
    def __init__(
        self,
        state_dimension: int,
        action_dimension: int,
        hidden_dimension: int,
        num_critics: int,
        num_quantiles: int,
    ):
        super().__init__()
        self.networks = torch.nn.ModuleList(
            [
                QuantileCriticModule(
                    state_dimension=state_dimension,
                    action_dimension=action_dimension,
                    hidden_dimension=hidden_dimension,
                    num_quantiles=num_quantiles,
                )
                for _ in range(num_critics)
            ]
        )
        self.targets = torch.nn.ModuleList(
            [
                QuantileCriticModule(
                    state_dimension=state_dimension,
                    action_dimension=action_dimension,
                    hidden_dimension=hidden_dimension,
                    num_quantiles=num_quantiles,
                )
                for _ in range(num_critics)
            ]
        )
        for network, target in zip(self.networks, self.targets):
            target.load_state_dict(network.state_dict())

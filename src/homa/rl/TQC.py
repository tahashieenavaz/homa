from .sac import SoftActor
from .tqc import QuantileMultiCritic


class TQC:
    def __init__(
        self,
        state_dimension: int,
        action_dimension: int,
        hidden_dimension: int = 256,
        actor_lr: float = 0.0003,
        actor_decay: float = 0.0,
        alpha: float = 0.2,
        min_std: float = -20.0,
        max_std: float = 2.0,
        num_quantiles: int = 25,
        num_critics: int = 5,
    ):
        self.critics = QuantileMultiCritic(
            num_critics=num_critics, num_quantiles=num_quantiles
        )
        self.actor = SoftActor(
            state_dimension=state_dimension,
            hidden_dimension=hidden_dimension,
            action_dimension=action_dimension,
            lr=actor_lr,
            weight_decay=actor_decay,
            alpha=alpha,
            min_std=min_std,
            max_std=max_std,
        )

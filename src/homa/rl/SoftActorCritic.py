from .sac import SoftActor, SoftCritic
from .buffers import ActorCriticBuffer


class SoftActorCritic:
    def __init__(
        self,
        state_dimension: int,
        action_dimension: int,
        hidden_dimension: int = 256,
        buffer_capacity: int = 1_000_000,
        batch_size: int = 256,
        actor_lr: float = 0.0002,
        critic_lr: float = 0.0003,
        actor_decay: float = 0.0,
        critic_decay: float = 0.0,
        tau: float = 0.005,
        gamma: float = 0.99,
        alpha: float = 0.2,
    ):
        self.batch_size: int = batch_size

        self.actor = SoftActor(
            state_dimension=state_dimension,
            action_dimension=action_dimension,
            hidden_dimension=hidden_dimension,
            lr=actor_lr,
            weight_decay=actor_decay,
        )
        self.critic = SoftCritic(
            state_dimension=state_dimension,
            action_dimension=action_dimension,
            hidden_dimension=hidden_dimension,
            lr=critic_lr,
            weight_decay=critic_decay,
        )
        self.buffer = ActorCriticBuffer(capacity=buffer_capacity)

    def train(self):
        data = self.buffer.sample(self.batch_size)
        self.critic.train()
        self.actor.train()

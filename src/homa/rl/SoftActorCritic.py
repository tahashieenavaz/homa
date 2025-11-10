from .sac import SoftActor, SoftCritic
from .buffers import SoftActorCriticBuffer


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
        alpha: float = 0.2,
        gamma: float = 0.99,
        min_std: float = -20,
        max_std: float = 2,
        warmup: int = 20_000,
    ):
        self.batch_size: int = batch_size
        self.warmup: int = warmup
        self.tau: float = tau

        self.actor = SoftActor(
            state_dimension=state_dimension,
            action_dimension=action_dimension,
            hidden_dimension=hidden_dimension,
            lr=actor_lr,
            weight_decay=actor_decay,
            alpha=alpha,
            min_std=min_std,
            max_std=max_std,
        )
        self.critic = SoftCritic(
            state_dimension=state_dimension,
            action_dimension=action_dimension,
            hidden_dimension=hidden_dimension,
            lr=critic_lr,
            weight_decay=critic_decay,
            gamma=gamma,
            alpha=alpha,
        )
        self.buffer = SoftActorCriticBuffer(capacity=buffer_capacity)

    def train(self):
        # don't train before warmup
        if self.buffer.size < self.warmup:
            return

        data = self.buffer.sample_torch(self.batch_size)
        self.critic.train(
            states=data.states,
            actions=data.actions,
            rewards=data.rewards,
            terminations=data.terminations,
            next_states=data.next_states,
            actor=self.actor,
        )
        self.actor.train(states=data.states, critic_network=self.critic.network)
        self.critic.update(tau=self.tau)

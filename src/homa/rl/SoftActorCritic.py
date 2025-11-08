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
        alpha: float = 0.2,
        gamma: float = 0.99,
        min_std: float = -20,
        max_std: float = 2,
    ):
        self.batch_size: int = batch_size

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
            tau=tau,
            gamma=gamma,
        )
        self.buffer = ActorCriticBuffer(capacity=buffer_capacity)

    def train(self):
        data = self.buffer.sample_torch(self.batch_size)
        self.critic.train(
            states=data.states,
            actions=data.actions,
            rewards=data.rewards,
            terminations=data.terminations,
            next_states=data.next_states,
            actor=self.actor.network,
        )
        self.actor.train(states=data.states, critic_network=self.critic.network)

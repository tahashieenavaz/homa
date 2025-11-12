from .sac import SoftActor, SoftCritic, SoftActorCriticTemperature
from .buffers import SoftActorCriticBuffer
from ..common.concerns import TracksTime
from ..device import get_device


class SoftActorCritic(TracksTime):
    def __init__(
        self,
        state_dimension: int,
        action_dimension: int,
        hidden_dimension: int = 256,
        buffer_capacity: int = 500_000,
        batch_size: int = 256,
        actor_lr: float = 0.0003,
        critic_lr: float = 0.0003,
        temperature_lr: float = 0.0003,
        actor_decay: float = 0.0,
        critic_decay: float = 0.0,
        temperature_decay: float = 0,
        tau: float = 0.005,
        gamma: float = 0.99,
        min_std: float = -20.0,
        max_std: float = 2.0,
        warmup: int = 20_000,
        device: None | str = None,
    ):
        super().__init__()

        self.batch_size: int = batch_size
        self.tau: float = tau
        self.warmup: int = warmup

        if device is None:
            device = get_device()

        self.actor = SoftActor(
            state_dimension=state_dimension,
            action_dimension=action_dimension,
            hidden_dimension=hidden_dimension,
            lr=actor_lr,
            weight_decay=actor_decay,
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
        )
        self.temperature = SoftActorCriticTemperature(
            lr=temperature_lr,
            weight_decay=temperature_decay,
            action_dimension=action_dimension,
            device=device,
        )
        self.buffer = SoftActorCriticBuffer(capacity=buffer_capacity)

    def is_warmup(self):
        return self.t < self.warmup

    def train(self):
        # don't train before warmup
        if self.is_warmup():
            return

        data = self.buffer.sample_torch(self.batch_size)
        alpha = self.temperature.alpha
        self.critic.train(
            states=data.states,
            actions=data.actions,
            rewards=data.rewards,
            terminations=data.terminations,
            next_states=data.next_states,
            actor=self.actor,
            alpha=alpha,
        )
        self.critic.update(tau=self.tau)
        self.actor.train(states=data.states, critic=self.critic, alpha=alpha)
        self.temperature.train()

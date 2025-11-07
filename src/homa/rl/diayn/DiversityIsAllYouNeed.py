from .Actor import Actor
from .Critic import Critic
from .Discriminator import Discriminator


class DiversityIsAllYouNeed:
    def __init__(
        self,
        state_dimension: int,
        action_dimension: int,
        hidden_dimension: int = 256,
        num_skills: int = 10,
        critic_decay: float = 0.0,
        actor_decay: float = 0.0,
        actor_lr: float = 0.0001,
        critic_lr: float = 0.001,
    ):
        self.actor = Actor(
            state_dimension=state_dimension,
            action_dimension=action_dimension,
            hidden_dimension=hidden_dimension,
            num_skills=num_skills,
            lr=actor_lr,
            decay=actor_decay,
        )
        self.critic = Critic(
            lr=critic_lr,
            num_skills=num_skills,
            hidden_dimension=hidden_dimension,
            decay=critic_decay,
        )
        self.discriminator = None

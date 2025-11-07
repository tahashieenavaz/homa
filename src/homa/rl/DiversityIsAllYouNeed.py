import torch
from .diayn.Actor import Actor
from .diayn.Critic import Critic
from .diayn.Discriminator import Discriminator
from .buffers import DiversityIsAllYouNeedBuffer, Buffer


class DiversityIsAllYouNeed:
    def __init__(
        self,
        state_dimension: int,
        action_dimension: int,
        hidden_dimension: int = 256,
        num_skills: int = 10,
        critic_decay: float = 0.0,
        actor_decay: float = 0.0,
        discriminator_decay: float = 0.0,
        actor_lr: float = 0.0001,
        critic_lr: float = 0.001,
        discriminator_lr=0.001,
        buffer_capacity: int = 1_000_000,
        actor_epsilon: float = 1e-6,
        gamma: float = 0.99,
    ):
        self.buffer: Buffer = DiversityIsAllYouNeedBuffer(capacity=buffer_capacity)
        self.num_skills: int = num_skills
        self.actor = Actor(
            state_dimension=state_dimension,
            action_dimension=action_dimension,
            hidden_dimension=hidden_dimension,
            num_skills=num_skills,
            lr=actor_lr,
            decay=actor_decay,
            epsilon=actor_epsilon,
        )
        self.critic = Critic(
            lr=critic_lr,
            num_skills=num_skills,
            hidden_dimension=hidden_dimension,
            decay=critic_decay,
            gamma=gamma,
        )
        self.discriminator = Discriminator(
            state_dimension=state_dimension,
            hidden_dimension=hidden_dimension,
            num_skills=num_skills,
            lr=discriminator_lr,
            decay=discriminator_decay,
        )

    def one_hot(indices, max_index):
        one_hot = torch.zeros(indices.size(0), max_index)
        one_hot.scatter_(1, indices.unsqueeze(1), 1)
        return one_hot

    def skill_index(self) -> torch.Tensor:
        return torch.randint(0, self.num_skills, (1,))

    def skill(self):
        return self.one_hot(self.skill_index(), self.num_skills)

    def train(self, skill: torch.Tensor):
        states, actions, rewards, next_states, terminations, log_probabilities = (
            self.buffer.all_tensor()
        )

        self.discriminator.train(states=states)
        self.critic.train()
        self.actor.train()

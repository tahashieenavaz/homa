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
        min_std: float = -20.0,
        max_std: float = 2.0,
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
            min_std=min_std,
            max_std=max_std,
        )
        self.critic = Critic(
            state_dimension=state_dimension,
            hidden_dimension=hidden_dimension,
            num_skills=num_skills,
            lr=critic_lr,
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

    def one_hot(self, indices, max_index) -> torch.Tensor:
        one_hot = torch.zeros(indices.size(0), max_index)
        one_hot.scatter_(1, indices.unsqueeze(1), 1)
        return one_hot

    def skill_index(self) -> torch.Tensor:
        return torch.randint(0, self.num_skills, (1,))

    def skill(self) -> torch.Tensor:
        return self.one_hot(self.skill_index(), self.num_skills)

    def advantages(
        self,
        states: torch.Tensor,
        skills: torch.Tensor,
        rewards: torch.Tensor,
        terminations: torch.Tensor,
        next_states: torch.Tensor,
    ) -> torch.Tensor:
        values = self.critic.values(states=states, skills=skills)
        termination_mask = 1 - terminations
        next_values = self.critic.values_(states=next_states, skills=skills)
        update = self.gamma * next_values * termination_mask
        return rewards + update - values

    def train(self, skill: torch.Tensor):
        data = self.buffer.all_tensor()
        skill_indices = skill.repeat(data.states.size(0), 1).long()
        skills_indices_one_hot = self.one_hot(skill_indices, self.num_skills)
        self.discriminator.train(
            states=data.states, skills_indices=skills_indices_one_hot
        )
        advantages = self.advantages(
            states=data.states,
            rewards=data.rewards,
            terminations=data.terminations,
            next_states=data.next_states,
            skills=skills,
        )
        self.critic.train(advantages=advantages)
        self.actor.train(advantages=advantages)

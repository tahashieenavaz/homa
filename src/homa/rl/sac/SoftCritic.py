import torch
from torch.nn.functional import mse_loss as mse
from .modules import DualSoftCriticModule
from .SoftActor import SoftActor
from ..utils import soft_update


class SoftCritic:
    def __init__(
        self,
        state_dimension: int,
        hidden_dimension: int,
        action_dimension: int,
        lr: float,
        weight_decay: float,
        gamma: float,
        alpha: float,
    ):
        self.gamma: float = gamma
        self.alpha: float = alpha

        self.network = DualSoftCriticModule(
            state_dimension=state_dimension,
            hidden_dimension=hidden_dimension,
            action_dimension=action_dimension,
        )
        self.target = DualSoftCriticModule(
            state_dimension=state_dimension,
            hidden_dimension=hidden_dimension,
            action_dimension=action_dimension,
        )

        # copy source to target when initiated
        self.target.load_state_dict(self.network.state_dict())

        self.optimizer = torch.optim.AdamW(
            self.network.parameters(), lr=lr, weight_decay=weight_decay
        )

    def train(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        terminations: torch.Tensor,
        next_states: torch.Tensor,
        actor: SoftActor,
    ):
        self.network.train()
        self.optimizer.zero_grad()
        loss = self.loss(
            states=states,
            actions=actions,
            rewards=rewards,
            terminations=terminations,
            next_states=next_states,
            actor=actor,
        )
        loss.backward()
        self.optimizer.step()

    def loss(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        terminations: torch.Tensor,
        next_states: torch.Tensor,
        actor: torch.nn.Module,
    ):
        q_alpha, q_beta = self.network(states, actions)
        target = self.calculate_target(
            rewards=rewards,
            terminations=terminations,
            next_states=next_states,
            actor=actor,
        )
        return mse(q_alpha, target) + mse(q_beta, target)

    @torch.no_grad()
    def calculate_target(
        self,
        rewards: torch.Tensor,
        terminations: torch.Tensor,
        next_states: torch.Tensor,
        actor: SoftActor,
    ):
        termination_mask = 1 - terminations
        next_actions, next_probabilities = actor.sample(next_states)
        q_alpha, q_beta = self.target(next_states, next_actions)
        q = torch.min(q_alpha, q_beta)
        entropy_q = q - self.alpha * next_probabilities
        return rewards + self.gamma * termination_mask * entropy_q

    def update(self, tau: float):
        soft_update(network=self.network.alpha, target=self.target.alpha, tau=tau)
        soft_update(network=self.network.beta, target=self.target.beta, tau=tau)

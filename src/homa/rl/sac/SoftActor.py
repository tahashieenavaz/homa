import torch
import numpy
from .modules import SoftActorModule


class SoftActor:
    def __init__(
        self,
        state_dimension: int,
        hidden_dimension: int,
        action_dimension: int,
        lr: float,
        weight_decay: float,
        alpha: float,
        min_std: float,
        max_std: float,
    ):
        self.alpha: float = alpha

        self.network = SoftActorModule(
            state_dimension=state_dimension,
            hidden_dimension=hidden_dimension,
            action_dimension=action_dimension,
            min_std=min_std,
            max_std=max_std,
        )
        self.optimizer = torch.optim.AdamW(
            self.network.parameters(), lr=lr, weight_decay=weight_decay
        )

    def train(self, states: torch.Tensor, critic_network: torch.nn.Module):
        self.network.train()
        self.optimizer.zero_grad()
        loss = self.loss(states=states, critic_network=critic_network)
        loss.backward()
        self.optimizer.step()

    def loss(
        self, states: torch.Tensor, critic_network: torch.nn.Module
    ) -> torch.Tensor:
        actions, probabilities = self.sample(states)
        q_alpha, q_beta = critic_network(states, actions)
        q = torch.min(q_alpha, q_beta)
        return (self.alpha * probabilities - q).mean()

    def process_state(self, state: numpy.ndarray | torch.Tensor) -> torch.Tensor:
        if isinstance(state, numpy.ndarray):
            state = torch.from_numpy(state).float()

        if state.ndim < 2:
            state = state.unsqueeze(0)

        return state

    def sample(self, state: numpy.ndarray | torch.Tensor):
        state = self.process_state(state)

        mean, std = self.network(state)
        # following line prevents standard deviations to be negative
        std = std.exp()

        distribution = torch.distributions.Normal(mean, std)

        pre_tanh = distribution.rsample()
        action = torch.tanh(pre_tanh)

        probabilities = distribution.log_prob(pre_tanh).sum(dim=1, keepdim=True)
        probabilities -= torch.log(1 - action.pow(2) + 1e-6).sum(dim=1, keepdim=True)

        return action, probabilities

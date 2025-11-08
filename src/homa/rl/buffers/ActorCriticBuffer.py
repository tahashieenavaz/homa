import numpy
import random
import torch
from .Buffer import Buffer


class ActorCriticBuffer(Buffer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def record(
        self,
        state: numpy.ndarray,
        action: int,
        reward: float,
        next_state: numpy.ndarray,
        termination: float,
        probability: numpy.ndarray,
    ):
        self.collection.append(
            (state, action, reward, next_state, termination, probability)
        )

    def sample(self, k: int, tensor: bool = False):
        batch = random.choice(self.collection, k=k)
        states, actions, rewards, next_states, terminations, probabilities = zip(*batch)

        states = numpy.array(states)
        actions = numpy.array(actions)
        rewards = numpy.array(rewards)
        next_states = numpy.array(next_states)
        terminations = numpy.array(terminations)
        probabilities = numpy.array(probabilities)

        if tensor:
            states = torch.from_numpy(states).float()
            actions = torch.from_numpy(actions).long()
            rewards = torch.from_numpy(rewards).float()
            next_states = torch.from_numpy(next_states).float()
            terminations = torch.from_numpy(terminations).float()
            probabilities = torch.from_numpy(probabilities).float()

        return states, actions, rewards, next_states, terminations, probabilities

    def sample_torch(self, k: int):
        return self.sample(k=k, torch=True)

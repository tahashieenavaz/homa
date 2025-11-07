import torch
import numpy
from types import SimpleNamespace
from .Buffer import Buffer
from .concerns import HasRecordAlternatives


class DiversityIsAllYouNeedBuffer(Buffer, HasRecordAlternatives):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def all_tensor(self) -> SimpleNamespace:
        return self.all(tensor=True)

    def all(self, tensor: bool = False) -> SimpleNamespace:
        states, actions, rewards, next_states, terminations, log_probabilities = zip(
            *self.collection
        )

        if tensor:
            states = torch.cat(states)
            actions = torch.cat(actions)
            rewards = torch.cat(rewards)
            next_states = torch.cat(next_states)
            terminations = torch.cat(terminations)
            log_probabilities = torch.cat(log_probabilities)

        return SimpleNamespace(
            **{
                "states": states,
                "actions": actions,
                "rewards": rewards,
                "next_states": next_states,
                "terminations": terminations,
                "log_probabilities": log_probabilities,
            }
        )

    def record(
        self,
        state: numpy.ndarray,
        action: int,
        reward: float,
        next_state: numpy.ndarray,
        termination: bool,
        log_probability: numpy.ndarray,
    ) -> None:
        self.collection.append(
            (state, action, reward, next_state, termination, log_probability)
        )

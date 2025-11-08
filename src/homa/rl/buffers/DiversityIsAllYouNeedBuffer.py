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
        states, actions, rewards, next_states, terminations, probabilities = zip(
            *self.collection
        )

        if tensor:
            states = torch.from_numpy(numpy.array(states))
            actions = torch.from_numpy(numpy.array(actions))
            rewards = torch.from_numpy(numpy.array(rewards))
            next_states = torch.from_numpy(numpy.array(next_states))
            terminations = torch.from_numpy(numpy.array(terminations))
            probabilities = torch.from_numpy(numpy.array(probabilities))

        return SimpleNamespace(
            **{
                "states": states,
                "actions": actions,
                "rewards": rewards,
                "next_states": next_states,
                "terminations": terminations,
                "probabilities": probabilities,
            }
        )

    def record(
        self,
        state: numpy.ndarray,
        action: int,
        reward: float,
        next_state: numpy.ndarray,
        termination: bool,
        probability: numpy.ndarray,
    ) -> None:
        self.collection.append(
            (state, action, reward, next_state, termination, probability)
        )

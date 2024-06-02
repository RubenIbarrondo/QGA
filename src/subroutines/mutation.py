import numpy as np
from abc import ABC, abstractmethod


class MutationSubroutine(ABC):

    @abstractmethod
    def mutate(self, state: np.ndarray) -> np.ndarray:
        pass

class RandomPauli(MutationSubroutine):

    def __init__(self, **kwargs):
        pass

    def mutate(self, state: np.ndarray) -> np.ndarray:
        return state
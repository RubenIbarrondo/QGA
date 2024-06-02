import numpy as np
from abc import ABC, abstractmethod

class CloningSubroutine(ABC):

    @abstractmethod
    def clone(self, state: np.ndarray) -> np.ndarray:
        return state

class UQCM(CloningSubroutine):

    def __init__(self, **kwargs):
        pass

    def clone(self, state: np.ndarray) -> np.ndarray:
        return state
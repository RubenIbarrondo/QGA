import numpy as np
from src.subroutines.abs_subroutine import AbstractSubroutine
from abc import abstractmethod

class MixingSubroutine(AbstractSubroutine):

    @abstractmethod
    def mix(self, state: np.ndarray) -> np.ndarray:
        pass

class MixingOff(MixingSubroutine):

    def mix(self, state: np.ndarray) -> np.ndarray:
        return state
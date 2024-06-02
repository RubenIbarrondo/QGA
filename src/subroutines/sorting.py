import numpy as np
from abc import ABC, abstractmethod

class SortingSubroutine(ABC):

    @abstractmethod
    def sort(self, state: np.ndarray) -> np.ndarray:
        pass

class FullSort(SortingSubroutine):
    
    def __init__(self, **kwargs):
        # define the sorting operation if needed
        pass

    def sort(self, state: np.ndarray) -> np.ndarray:
        return state

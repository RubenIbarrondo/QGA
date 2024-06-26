import numpy as np
from qga_toolbox.subroutines.abs_subroutine import AbstractSubroutine

from pyqch.state_transformations import subsystem_permutation
from abc import abstractmethod

class MixingSubroutine(AbstractSubroutine):

    @abstractmethod
    def mix(self, state: np.ndarray) -> np.ndarray:
        pass

class MixingOff(MixingSubroutine):

    def mix(self, state: np.ndarray) -> np.ndarray:
        return state
    

class MixingFixedIndex(MixingSubroutine):

    def __init__(self, chromosome_size: int, population_size: int, mixing_index: int, **kwargs):
        super().__init__(chromosome_size, population_size, **kwargs)

        if mixing_index > chromosome_size +1:
            raise ValueError(f"Mixing index must be between 0 and chromosome_size, but {mixing_index} > {chromosome_size}")
        
        self.mixing_index = mixing_index
        self.system_shape_internal = tuple([2 ** self.mixing_index, 2 ** (self.chromosome_size - self.mixing_index)] * self.population_size)
        
        self.mixing_permutation = self._get_mixing_permutation()

    def _get_mixing_permutation(self):
        mp = list(range(len(self.system_shape_internal)))
        for j in range(self.population_size // 4):
            mp[self.population_size + 4 * j + 1] = self.population_size + 4 * j + 3
            mp[self.population_size + 4 * j + 3] = self.population_size + 4 * j + 1
        return tuple(mp)

    def mix(self, state: np.ndarray) -> np.ndarray:
        return subsystem_permutation(state, self.system_shape_internal, self.mixing_permutation)
        


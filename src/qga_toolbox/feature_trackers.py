import numpy as np
from abc import ABC, abstractmethod

from pyqch import state_transformations

class FeatureTracker(ABC):

    def __init__(self, chromosome_size: int, population_size: int, **kwargs):
        self.chromosome_size = chromosome_size
        self.population_size = population_size
        self.dim = 2 ** self.chromosome_size
        self.dim_pop = self.dim ** self.population_size
        self.system_shape = (self.dim,) * self.population_size

    def set_up(self, **kwargs):
        pass

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            for key in self.__dict__.keys():
                if key not in other.__dict__:
                    return False
                try:
                    if not self.__dict__[key] == other.__dict__[key]:
                        return False
                except ValueError as verr:
                    try:
                        if not all(self.__dict__[key] == other.__dict__[key]):
                            return False
                    except ValueError as verr2:
                        if not np.all(self.__dict__[key] == other.__dict__[key]):
                            return False
            return True
        else:
            return False

    @abstractmethod
    def track(self, state: np.ndarray) -> np.ndarray:
        pass


class IndividualEigenstateFidelity(FeatureTracker):

    def __init__(self, chromosome_size: int, population_size: int, energy_basis: np.ndarray = None, **kwargs):
        super().__init__(chromosome_size=chromosome_size, population_size=population_size, **kwargs)

        self.set_up(energy_basis)            
    
    def set_up(self, energy_basis: np.ndarray = None):
        if energy_basis is None:
            self.energy_basis = None
        else:
            if energy_basis.shape[0] != self.dim:
                raise ValueError(f"Incompatible energy_basis with shape {energy_basis.shape} for self dimension {self.dim}.")
            self.energy_basis = energy_basis


    def track(self, state: np.ndarray) -> np.ndarray:
        if self.energy_basis is None:
            return np.trace(state)
        
        fidelity_matrix = np.zeros((self.population_size, self.dim))

        for i_reg in range(self.population_size):
            all_except_i = tuple(list(range(i_reg))+list(range(i_reg+1,self.population_size)))
            state_reg = state_transformations.partial_trace(state, self.system_shape, all_except_i)

            for i_eig, eigenstate in enumerate(self.energy_basis):
                fidelity_matrix[i_reg, i_eig] = np.real(eigenstate.T.conj() @ state_reg @ eigenstate)

        return fidelity_matrix
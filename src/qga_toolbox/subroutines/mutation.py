import numpy as np
from qga_toolbox.subroutines.abs_subroutine import AbstractSubroutine
from abc import abstractmethod

from pyqch.channel_families import probabilistic_unitaries
from pyqch import state_transformations
from pyqch import random_generators

class MutationSubroutine(AbstractSubroutine):

    @abstractmethod
    def mutate(self, state: np.ndarray) -> np.ndarray:
        pass

class RandomPauli(MutationSubroutine):

    def __init__(self, chromosome_size: int, population_size: int, mutation_probability: float, random_state: int | np.random.Generator = None, **kwargs):
        if random_state is not None:
            self.random_state = np.random.default_rng(random_state)
        else:
            # As the methods in random use the default random generator
            # we access it via the __self__ attribute
            self.random_state = np.random.random.__self__

        self.chromosome_size = chromosome_size
        self.population_size = population_size
        self.mutation_probability = mutation_probability
        self.system_shape = (2,) * (self.chromosome_size * self.population_size)

        # Denfine the Pauli matrices
        paulix = np.zeros((2, 2))
        paulix[0,1] = 1
        paulix[1,0] = 1

        pauliy = np.zeros((2, 2), dtype=complex)
        pauliy[0,1] = -1j
        pauliy[1,0] = 1j

        pauliz = np.diag([1,-1])
        
        # Define the channels for applying the Pauli operators
        self.x = np.kron(paulix, paulix)
        self.y = np.kron(pauliy, pauliy.T)
        self.z = np.kron(pauliz, pauliz)
        self.unitaries_aschannel_array = np.array([np.identity(4), self.x, self.y, self.z])

        # Define the channel for applying a random Pauli operator
        self.probability_array = np.array([1-self.mutation_probability,
                                      self.mutation_probability/3,
                                      self.mutation_probability/3,
                                      self.mutation_probability/3])
        self.unitaries_array = np.array([np.identity(2), paulix, pauliy, pauliz])
        self.local_mutation_mat = probabilistic_unitaries(2,
                                                          self.probability_array,
                                                          self.unitaries_array)
                                                          

    def mutate(self, state: np.ndarray, random_select: bool = False) -> np.ndarray:
        state_mut = state.copy()

        if random_select:
            mutation_pattern = self.random_state.choice(4,
                                                        self.chromosome_size * self.population_size,
                                                        p = self.probability_array)
            for chromosome, unitary_index in enumerate(mutation_pattern):
                if unitary_index > 0:
                    paulit = self.unitaries_aschannel_array[unitary_index]
                    state_mut = state_transformations.local_channel(state_mut,
                                                                    self.system_shape,
                                                                    chromosome,
                                                                    paulit)
        else:

            for chromosome in range(self.chromosome_size * self.population_size):
                state_mut = state_transformations.local_channel(state_mut,
                                                                self.system_shape,
                                                                chromosome,
                                                                self.local_mutation_mat)
        return state_mut
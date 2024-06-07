import numpy as np
from abc import ABC, abstractmethod
from typing import Iterable

from pyqch import state_transformations

class SortingSubroutine(ABC):

    def __init__(self, chromosome_size: int, population_size: int, **kwargs):
        self.chromosome_size = chromosome_size
        self.population_size = population_size
        self.dim = 2 ** self.chromosome_size
        self.dim_pop = self.dim ** self.population_size

        self.system_shape = (self.dim,) * self.population_size

    @abstractmethod
    def sort(self, state: np.ndarray) -> np.ndarray:
        # Also give the choice to do a probabilistic sort
        # in the sense that the output is one of the sorted populations
        # oh, what about pure states or states of limited rank?
        pass

class SortingNetwork(SortingSubroutine):

    @property
    @abstractmethod
    def depth(self) -> int:
        return None

    @property
    @abstractmethod
    def network(self) -> Iterable[list[tuple]]:
        # Encodes  a network in layers and pairs
        # Returns an iterable with list of tuples of pairs
        for layer in range(self.depth):
            yield layer

    @abstractmethod
    def pairwise_sort(self, state: np.ndarray, i_first: int, i_second: int) -> np.ndarray:
        # Returns a version of the state with the positions i_firtst and i_second sorted
        # The notion of "sorting" can be different depending on the implementation
        return state

    def sort(self, state: np.ndarray) -> np.ndarray:
        state_sort = state.copy()
        for layer in self.network:
            for i,j in layer:
                state_sort = self.pairwise_sort(state_sort,
                                                i_first=i,
                                                i_second=j)
        return state

class FullSort(SortingNetwork):
    
    def __init__(self,  chromosome_size: int, population_size: int, hamiltonian: np.ndarray, degtol: float = 1e-6, **kwargs):
        super().__init__( chromosome_size, population_size, **kwargs)

        # Parse information about the Hamiltonian
        self.hamiltonian = hamiltonian
        w, u = np.linalg.eigh(self.hamiltonian)
        self.basis = u
        self.spec = w

        # Check whether it is degenerate up to degtol
        self.degtol = degtol
        self.is_degenerate = np.any(np.diff(self.spec) <= self.degtol)

        self.pairwise_sort_mat = self._pairwise_sort_mat()
    
    def _pairwise_sort_mat(self):
        if self.is_degenerate:
            # Create a degeneracy flag and use that in the iterators...
            # Maybe with an einsum everything is quite simple (ege have a degeneracy symmetric matrix with ones and zeros..., this would be the identity if non deg)
            raise NotImplementedError("Degenerate Hamiltonians are not implemented yet.")
        
        # In the computational basis
        a0 = np.zeros((self.dim, self.dim, self.dim, self.dim))
        for i in range(self.dim):
            for j in range(i, self.dim):
                a0[i,j,i,j] = 1

        a1 = np.zeros((self.dim, self.dim, self.dim, self.dim))
        for i in range(self.dim):
            for j in range(i):
                a1[j,i,i,j] = 1
        
        # Rotate back to the problem basis and put in matrix form
        a0 = np.einsum("ij,kl,jlst->ikst", self.basis, self.basis, a0)
        a0 = a0.reshape((self.dim**2, self.dim**2))
        a1 = np.einsum("ij,kl,jlst->ikst", self.basis, self.basis, a1)
        a1 = a1.reshape((self.dim**2, self.dim**2))
        
        # If this is true, then the channel should be trace preserving!!!
        #print("Gauge condition", np.allclose(a0.T.conj() @ a0 + a1.T.conj() @ a1, np.identity(self.dim ** 2)))
        
        tmat = np.kron(a0,a0.T) + np.kron(a1,a1.T)

        # Also...
        # First a PVM checking whether they are sorted or not
        # Then apply a cswap, need to define the cswap matrix
        return tmat
    
    @property
    def depth(self):
        return self.population_size
    
    @property
    def network(self) -> Iterable[list[tuple]]:
        even = [(i,i+1) for i in range(0, self.population_size-1, 2)]
        odd = [(i,i+1) for i in range(1, self.population_size-1, 2)]
        layer_choice = [even, odd]
        
        for i_layer in range(self.depth):
            yield layer_choice[i_layer % 2]


    def pairwise_sort(self, state: np.ndarray, i_first: int, i_second: int) -> np.ndarray:
        return state_transformations.local_channel(state,
                                                   self.system_shape,
                                                   (i_first, i_second),
                                                   self.pairwise_sort_mat)

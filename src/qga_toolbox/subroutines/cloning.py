import numpy as np
from qga_toolbox.subroutines.abs_subroutine import AbstractSubroutine
from abc import abstractmethod

from pyqch.predicates import is_system_compatible
import pyqch.state_transformations as st
import pyqch.channel_families as cf
import pyqch.channel_operations as co

class CloningSubroutine(AbstractSubroutine):

    @abstractmethod
    def clone(self, state: np.ndarray, single_individual: bool = False) -> np.ndarray:
        return state

class PairWiseClone(CloningSubroutine):

    def __init__(self, chromosome_size: int, population_size: int, clone_mat: np.ndarray, **kwargs):
        super().__init__(chromosome_size, population_size, **kwargs)
        self.n = int(population_size)

        self.clone_mat = clone_mat

    def clone(self, state: np.ndarray, single_individual: bool = False) -> np.ndarray:
        if single_individual:
            return (self.clone_mat @ state.reshape(self.dim ** 2)).reshape((self.dim ** 2, self.dim ** 2))

        n_current = self.n // 2

        if not is_system_compatible(state, (self.dim,) * n_current):
            raise ValueError(f"State with shape {state.shape} cannot be interpreted as system shape ({self.dim},)*{n_current}.")
        
        for individual in range(self.n // 2):
            
            state = st.local_channel(state, (self.dim,) * n_current, individual, self.clone_mat)
            n_current += 1
            perm = list(range(n_current))
            perm[individual+1] = n_current
            perm[individual+1:] = [p-1 for p in perm[individual+1:]]
            state = st.subsystem_permutation(state, (self.dim,) * n_current, permutation=tuple(perm))

        return state

class UQCM(PairWiseClone):

    def __init__(self, chromosome_size: int, population_size: int, **kwargs):
        self.dim = 2 ** chromosome_size

        # Build the projector onto the symmetric subspace of 2 individuals
        id2 = np.identity(self.dim ** 2)
        swap = np.einsum('ijkl->jikl', id2.reshape((self.dim,) * 4)).reshape((self.dim ** 2, self.dim ** 2))
        sym_proj = .5 * (swap + id2)
        sym_proj = np.kron(sym_proj, sym_proj)

        # Build the extending with identity operation
        states = (np.identity(self.dim) / self.dim).reshape((1, self.dim, self.dim))
        id_extend = co.tensor([np.identity(self.dim**2),
                               cf.initializer(self.dim, states)])
        
        # Build 1 to 2 cloning matrix
        clone_mat = 2 * self.dim / (self.dim + 1) * sym_proj @ id_extend

        super().__init__(chromosome_size, population_size, clone_mat=clone_mat, **kwargs)

class BCQO(PairWiseClone):

    def __init__(self, chromosome_size: int, population_size: int, cloning_basis: int | np.ndarray, **kwargs):
        
        self.dim = 2 ** chromosome_size

        self.cloning_basis = cloning_basis

        # Build 1 to 2 cloning matrix in computational basis
        clone_isometry_computational = np.zeros((self.dim,) * (2+1))
        for i in range(self.dim):
            clone_isometry_computational[i,i,i]=1
        clone_isometry_computational = clone_isometry_computational.reshape((self.dim**2, self.dim))
        
        if isinstance(self.cloning_basis, int) and self.cloning_basis == 1:
            clone_isometry = clone_isometry_computational
        elif isinstance(self.cloning_basis, np.ndarray):
            clone_isometry = np.dot(np.kron(self.cloning_basis.T.conj(), self.cloning_basis.T.conj()),
                                    np.dot(clone_isometry_computational,
                                           self.cloning_basis))
        else:
            raise ValueError(f"Unexpected value for cloning_basis, expected 1 or np.ndarray but obtained:\n{cloning_basis}.")
        
        clone_mat = np.kron(clone_isometry, clone_isometry.conj())

        super().__init__(chromosome_size, population_size, clone_mat=clone_mat, **kwargs)
    

if __name__ == "__main__":
    c = 2
    dim = 2 ** c
    n = 4
    uq = UQCM(c, n)
    state = np.identity(dim ** (n// 2)) / (dim ** (n//2))
    state2 = uq.clone(state)
    print(state2.shape)
import numpy as np
from abc import ABC, abstractmethod

import pyqch.state_transformations as st
import pyqch.channel_families as cf
import pyqch.channel_operations as co

class CloningSubroutine(ABC):

    @abstractmethod
    def clone(self, state: np.ndarray) -> np.ndarray:
        return state

class UQCM(CloningSubroutine):

    def __init__(self, chromosome_size: int, population_size: int, **kwargs):

        self.dim = int(2 ** chromosome_size)
        self.n = int(population_size)

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
        self.clone_mat = 2 * self.dim / (self.dim + 1) * sym_proj @ id_extend

    def clone(self, state: np.ndarray) -> np.ndarray:
        
        n_current = self.n // 2
        
        for individual in range(self.n // 2):
            
            state = st.local_channel(state, (self.dim,) * n_current, individual, self.clone_mat)
            n_current += 1
            perm = list(range(n_current))
            perm[individual+1] = n_current
            perm[individual+1:] = [p-1 for p in perm[individual+1:]]
            state = st.subsystem_permutation(state, (self.dim,) * n_current, permutation=tuple(perm))

        return state
    

if __name__ == "__main__":
    c = 2
    dim = 2 ** c
    n = 4
    uq = UQCM(c, n)
    state = np.identity(dim ** (n// 2)) / (dim ** (n//2))
    state2 = uq.clone(state)
    print(state2.shape)
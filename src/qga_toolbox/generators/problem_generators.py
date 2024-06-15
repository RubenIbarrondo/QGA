from typing import Iterable, Literal
import numpy as np
from scipy.stats import unitary_group

class _ProblemGenerator:

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

class HaarRandomHamiltonian(_ProblemGenerator):

    def __init__(self, chromosome_size: int, energies: Iterable | Literal['non-degenerate'], problem_instance_number: int, seed: int | np.random.Generator, **kwargs):
        self.chromosome_size = chromosome_size
        self.dimension = 2 ** self.chromosome_size

        self.problem_instance_number = problem_instance_number
        if energies == 'non-degenerate':
            self.energies = np.arange(self.dimension)
        else:
            self.energies = np.array(energies)
    
    def generate(self):
        ref_hamiltonian = np.diag(self.energies)
        for _ in range(self.problem_instance_number):
            u = unitary_group(self.dimension)
            problem_hamiltonian = u @ ref_hamiltonian @ u.T.conj()
            yield problem_hamiltonian

class HamiltonianSampleDirectory(_ProblemGenerator):

    def __init__(self, dirpath: str, **kwargs):
        self.dirpath = dirpath
        
    def generate(self):
        # This is just a stand-in function
        for pi in range(10):
            problem_hamiltonian = np.diag(np.arange(4))
            yield problem_hamiltonian
from typing import Iterable, Literal
import numpy as np

def haar_random_hamiltonian(chromosome_size: int, energies: Iterable | Literal['non-degenerate'], problem_instance_number: int, seed: int | np.random.Generator, **kwargs):
    dimension = 2 ** chromosome_size
    for pi in range(problem_instance_number):
        problem_hamiltonian = np.diag(np.arange(dimension))
        yield problem_hamiltonian

def hamiltonian_sample_directory(dirpath: str, **kwargs):
    
    for pi in range(10):
        problem_hamiltonian = np.diag(np.arange(4))
        yield problem_hamiltonian
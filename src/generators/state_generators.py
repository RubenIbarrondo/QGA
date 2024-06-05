import numpy as np

def haar_random_states(population_size: int, chromosome_size: int, initial_population_number: int, seed: int | np.random.Generator, **kwargs):

    for ip in range(initial_population_number):
        initial_state = np.identity(2**(population_size * chromosome_size)) / 2**(population_size * chromosome_size)
        yield initial_state

def init_state_sample_directory(dirpath: str, **kwargs):
    
    for ip in range(10):
        initial_state = np.identity(2**(4 * 2)) / 2**(4 * 2)
        yield initial_state
import numpy as np


class HaarRandomStates:

    def __init__(self, population_size: int, chromosome_size: int, initial_population_number: int, seed: int | np.random.Generator, **kwargs):
        self.initial_population_number = initial_population_number
        self.population_size = population_size
        self.chromosome_size = chromosome_size

    def generate(self):
        for ip in range(self.initial_population_number):
            initial_state = np.identity(2**(self.population_size * self.chromosome_size)) / 2**(self.population_size * self.chromosome_size)
            yield initial_state

class InitStateSampleDirectory:

    def __init__(self, dirpath: str, **kwargs):
        self.dirpath = dirpath
        pass

    def generate(self):
        # Just a stand-in function
        for ip in range(10):
            initial_state = np.identity(2**(4 * 2)) / 2**(4 * 2)
            yield initial_state
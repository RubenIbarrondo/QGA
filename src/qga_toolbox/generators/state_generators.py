import numpy as np

from pyqch import random_generators

class _StateGenerator:

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

class HaarRandomStates(_StateGenerator):

    def __init__(self, population_size: int, chromosome_size: int, initial_population_number: int, seed: int | np.random.Generator, **kwargs):
        self.initial_population_number = initial_population_number
        self.population_size = population_size
        self.chromosome_size = chromosome_size
        
        self.seed = seed
        
    def generate(self):
        random_state_generator = np.random.default_rng(self.seed)
        for ip in range(self.initial_population_number):
            initial_state = random_generators.state(2**(self.population_size * self.chromosome_size),
                                                    rank=1,
                                                    random_state=random_state_generator)
            yield initial_state

class InitStateSampleDirectory(_StateGenerator):

    def __init__(self, dirpath: str, **kwargs):
        self.dirpath = dirpath
        pass

    def generate(self):
        # Just a stand-in function
        for ip in range(10):
            initial_state = np.identity(2**(4 * 2)) / 2**(4 * 2)
            yield initial_state
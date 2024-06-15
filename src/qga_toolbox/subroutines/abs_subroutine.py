from abc import ABC
import numpy as np

class AbstractSubroutine(ABC):

    def __init__(self, chromosome_size: int, population_size: int, **kwargs):
        self.chromosome_size = chromosome_size
        self.population_size = population_size
        self.dim = 2 ** self.chromosome_size
        self.dim_pop = self.dim ** self.population_size

        self.system_shape = (self.dim,) * self.population_size
        

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
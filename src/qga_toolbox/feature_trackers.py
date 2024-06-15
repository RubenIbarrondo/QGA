import numpy as np
from abc import ABC, abstractmethod

class FeatureTracker(ABC):

    def __init__(self, **kwargs):
        pass

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

    @abstractmethod
    def track(self, state: np.ndarray) -> np.ndarray:
        pass


class IndividualEigenstateFidelity(FeatureTracker):

    def track(self, state: np.ndarray) -> np.ndarray:
        return np.array([np.trace(state)])
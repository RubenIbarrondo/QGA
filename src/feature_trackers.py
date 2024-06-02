import numpy as np
from abc import ABC, abstractmethod

class FeatureTracker(ABC):

    @abstractmethod
    def track(self, state: np.ndarray) -> np.ndarray:
        pass


class IndividualEigenstateFidelity(FeatureTracker):

    def track(self, state: np.ndarray) -> np.ndarray:
        return np.array([np.trace(state)])
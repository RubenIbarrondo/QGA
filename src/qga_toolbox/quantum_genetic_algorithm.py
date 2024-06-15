import pyqch.state_transformations
from qga_toolbox.subroutines.cloning import CloningSubroutine
from qga_toolbox.subroutines.sorting import SortingSubroutine
from qga_toolbox.subroutines.mixing import MixingSubroutine
from qga_toolbox.subroutines.mutation import MutationSubroutine
from qga_toolbox.feature_trackers import FeatureTracker

import numpy as np
import pyqch

def quantum_genetic_algorithm(initial_state: np.ndarray,
                              cloning: CloningSubroutine,
                              sorting: SortingSubroutine,
                              mixing: MixingSubroutine,
                              mutation: MutationSubroutine,
                              generations: int,
                              population_size: int,
                              chromosome_size: int,
                               track_features: dict[str, FeatureTracker] = dict(),
                              **kwargs) -> tuple[np.ndarray, dict[str,list[np.ndarray]]]:

    state_offspring = initial_state
    tracked_features = {feature_name: [feature_tracker.track(state_offspring)] for feature_name, feature_tracker in track_features.items()}

    def _record_feature(state):
        for feature_name, feature_tracker in track_features.items():
            tracked_features[feature_name].append(feature_tracker.track(state))

    for generation in range(generations):

        state_sorted = sorting.sort(state_offspring)
        _record_feature(state_sorted)

        state_selected = pyqch.state_transformations.partial_trace(state_sorted,
                                                                   (2 ** chromosome_size,) * population_size,
                                                                   tuple(range(population_size//2, population_size)))

        state_cloned = cloning.clone(state_selected)
        _record_feature(state_cloned)

        state_mixed = mixing.mix(state_cloned)
        _record_feature(state_mixed)

        state_mutation = mutation.mutate(state_mixed)
        state_offspring = state_mutation
        _record_feature(state_offspring)

    return state_offspring, tracked_features
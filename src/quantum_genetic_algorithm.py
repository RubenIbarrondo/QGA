from subroutines.cloning import CloningSubroutine
from subroutines.sorting import SortingSubroutine
from subroutines.mutation import MutationSubroutine
from feature_trackers import FeatureTracker
import numpy as np

def quantum_genetic_algorithm(initial_state: np.ndarray,
                              cloning: CloningSubroutine,
                              sorting: SortingSubroutine,
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
        
        state_selected = sorting.sort(state_offspring)
        _record_feature(state_selected)

        state_cloned = cloning.clone(state_selected)
        _record_feature(state_cloned)

        state_mutation = mutation.mutate(state_cloned)
        state_offspring = state_mutation
        _record_feature(state_offspring)

    return state_offspring, tracked_features
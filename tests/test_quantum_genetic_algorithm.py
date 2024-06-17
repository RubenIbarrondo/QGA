import unittest

import numpy as np

from qga_toolbox.quantum_genetic_algorithm import quantum_genetic_algorithm
from qga_toolbox.subroutines import cloning, sorting, mixing, mutation

class TestQuantumGeneticAlgorithm(unittest.TestCase):

    def test_uqcm_mixingoff_canonicalH(self) -> None:
        ps = 4
        cs = 2
        g = 1
        state = np.identity(2 ** (ps * cs)) / (2 ** (ps * cs))
        hc = np.diag(np.arange(2 ** cs))
        mprob = .1

        c = cloning.UQCM(population_size = ps, chromosome_size = cs)
        s = sorting.FullSort(population_size = ps, chromosome_size = cs, hamiltonian=hc)
        mx = mixing.MixingOff(population_size = ps, chromosome_size = cs)
        mt = mutation.RandomPauli(population_size = ps, chromosome_size = cs, mutation_probability=mprob)

        tfs = {}
        final_state, tracked = quantum_genetic_algorithm(initial_state= state,
                                                        cloning = c,
                                                        sorting = s,
                                                        mixing = mx,
                                                        mutation = mt,
                                                        generations = g,
                                                        population_size = ps,
                                                        chromosome_size = cs,
                                                        track_features=tfs )
        
        # Test shapes
        self.assertEqual(tracked, {})
        self.assertEqual(final_state.shape, state.shape)

        # Test expected results


if __name__ == '__main__':
    ps = 4
    cs = 2
    g = 1
    state = np.identity(2 ** (ps * cs)) / (2 ** (ps * cs))
    hc = np.diag(np.arange(2 ** cs))
    mprob = .1

    c = cloning.UQCM(population_size = ps, chromosome_size = cs)
    s = sorting.FullSort(population_size = ps, chromosome_size = cs, hamiltonian=hc)
    mx = mixing.MixingOff(population_size = ps, chromosome_size = cs)
    mt = mutation.RandomPauli(population_size = ps, chromosome_size = cs, mutation_probability=mprob)

    tfs = {}
    final_state, tracked = quantum_genetic_algorithm(initial_state= state,
                                                    cloning = c,
                                                    sorting = s,
                                                    mixing = mx,
                                                    mutation = mt,
                                                    generations = g,
                                                    population_size = ps,
                                                    chromosome_size = cs,
                                                    track_features=tfs )
    print(final_state.shape)
    print(tracked)
import unittest

import numpy as np

from qga_toolbox.quantum_genetic_algorithm import quantum_genetic_algorithm
from qga_toolbox.subroutines import cloning, sorting, mixing, mutation
from qga_toolbox import feature_trackers

class TestQuantumGeneticAlgorithm(unittest.TestCase):

    def test_clone_and_conquer(self) -> None:
        ps = 2
        cs = 2
        g = 5

        state = np.identity(2 ** (ps * cs)) / (2 ** (ps * cs))
        basis = np.identity(2 ** cs)
        hc = basis @ np.diag(np.arange(2 ** cs)) @ basis.T.conj()
        mprob = 0.0

        c = cloning.UQCM(population_size = ps, chromosome_size = cs)
        s = sorting.FullSort(population_size = ps, chromosome_size = cs, hamiltonian=hc)
        mx = mixing.MixingOff(population_size = ps, chromosome_size = cs)
        mt = mutation.RandomPauli(population_size = ps, chromosome_size = cs, mutation_probability=mprob)

        fid = "fidelities"
        tfs = {fid: feature_trackers.IndividualEigenstateFidelity(chromosome_size=cs,population_size=ps,energy_basis=basis)}
        final_state, tracked = quantum_genetic_algorithm(initial_state= state,
                                                        cloning = c,
                                                        sorting = s,
                                                        mixing = mx,
                                                        mutation = mt,
                                                        generations = g,
                                                        population_size = ps,
                                                        chromosome_size = cs,
                                                        track_features=tfs)
        
        # Test shapes
        self.assertTrue(isinstance(tracked,dict))
        self.assertEqual(final_state.shape, state.shape)

        # Test expected results
        print(np.array(tracked[fid]))
        print()
        first_fid_post_sort = np.array(tracked[fid])[::4, 0, 0]  # Fidelity of the first register with the ground state right after sort
        fid_ref = 1 - (2 ** cs / (2 ** cs+1)) ** (ps / 2 * np.arange(g+1)) * (1 - 1/ 2 ** cs)
        np.testing.assert_array_almost_equal(first_fid_post_sort, fid_ref)

    def test_uqcm_mixingoff_canonicalH(self) -> None:
        ps = 4
        cs = 2
        g = 5
        state = np.identity(2 ** (ps * cs)) / (2 ** (ps * cs))
        basis = np.identity(2 ** cs)
        hc = basis @ np.diag(np.arange(2 ** cs)) @ basis.T.conj()
        mprob = 0.0

        c = cloning.UQCM(population_size = ps, chromosome_size = cs)
        s = sorting.FullSort(population_size = ps, chromosome_size = cs, hamiltonian=hc)
        mx = mixing.MixingOff(population_size = ps, chromosome_size = cs)
        mt = mutation.RandomPauli(population_size = ps, chromosome_size = cs, mutation_probability=mprob)

        fid = "fidelities"
        tfs = {fid: feature_trackers.IndividualEigenstateFidelity(chromosome_size=cs,population_size=ps,energy_basis=basis)}
        final_state, tracked = quantum_genetic_algorithm(initial_state= state,
                                                        cloning = c,
                                                        sorting = s,
                                                        mixing = mx,
                                                        mutation = mt,
                                                        generations = g,
                                                        population_size = ps,
                                                        chromosome_size = cs,
                                                        track_features=tfs)
        
        # Test shapes
        self.assertTrue(isinstance(tracked,dict))
        self.assertEqual(final_state.shape, state.shape)

        # Test expected results
        first_fid_post_sort = np.array(tracked[fid])[::4, 0, 0]  # Fidelity of the first register with the ground state right after sort
        fid_ref = 1 - (2 ** cs / (2 ** cs+1)) ** (ps / 2 * np.arange(g+1)) * (1 - 1/ 2 ** cs)
        np.testing.assert_array_almost_equal(first_fid_post_sort, fid_ref)


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
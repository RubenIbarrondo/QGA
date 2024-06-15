import unittest

import numpy as np
from pyqch import predicates
from pyqch import channel_operations
from pyqch import state_transformations
from pyqch import channel_families
from pyqch import random_generators
from qga_toolbox.subroutines import mutation

class TestSubroutinesMutation_RandomPauli(unittest.TestCase):

    def setUp(self) -> None:
        self.chromosome_size = 2
        self.population_size = 2
        self.mutation_probability = .1

        self.system_shape = (2,) * (self.chromosome_size * self.population_size)
        self.system_dim = 2 ** (self.chromosome_size * self.population_size)
        self.random_state = 24597856786290

        self.rpauli = mutation.RandomPauli(self.chromosome_size, self.population_size, self.mutation_probability)

        self.pop_state = np.zeros((self.system_dim, self.system_dim))
        self.pop_state[0,0] = 1

        self.arbpsi = random_generators.state(self.system_dim, 1, self.random_state)
    
    def test_shapes(self):
        for u_aschannel in self.rpauli.unitaries_aschannel_array:
            self.assertEqual(u_aschannel.shape, (4,4))
        for u in self.rpauli.unitaries_array:
            self.assertEqual(u.shape, (2,2))
        
        pop_state2 = self.rpauli.mutate(self.pop_state)
        self.assertEqual(pop_state2.shape, self.pop_state.shape)

        pop_state3 = self.rpauli.mutate(self.pop_state, random_select=True)
        self.assertEqual(pop_state3.shape, self.pop_state.shape)

    def test_check_paulis(self):

        paulis = [np.array([[1,0],[0,1]]),
                  np.array([[0,1],[1,0]]),
                  np.array([[0,-1j],[1j,0]]),
                  np.array([[1,0],[0,-1]])]

        for i, u in enumerate(self.rpauli.unitaries_array):
            np.testing.assert_array_almost_equal(u, paulis[i])

    def test_is_channel(self):
        for u_aschannel in self.rpauli.unitaries_aschannel_array:
            self.assertTrue(predicates.is_channel(u_aschannel))
        for u in self.rpauli.unitaries_array:
            np.testing.assert_array_almost_equal(u @ u.T.conj(), np.identity(2))

    def test_unitality(self):
        maxmix = np.identity(self.system_dim)

        maxmix2 = self.rpauli.mutate(maxmix)
        maxmix3 = self.rpauli.mutate(maxmix, random_select=True)

        np.testing.assert_array_almost_equal(maxmix2, maxmix)
        np.testing.assert_array_almost_equal(maxmix3, maxmix)
        
    def test_probzero_isidentity(self):
        zeromut = mutation.RandomPauli(self.chromosome_size, self.population_size, 0.0)
        arbpsi2 = zeromut.mutate(self.arbpsi)
        np.testing.assert_array_almost_equal(arbpsi2, self.arbpsi)

        arbpsi3 = zeromut.mutate(self.arbpsi, random_select=True)
        np.testing.assert_array_almost_equal(arbpsi3, self.arbpsi)
    
    def test_prob34_isreplacer(self):
        ppauli_from_depol = 3/4
        treplacer = channel_families.depolarizing(2, 1.0)
        onemut = mutation.RandomPauli(self.chromosome_size, self.population_size, ppauli_from_depol)
        np.testing.assert_array_almost_equal(onemut.local_mutation_mat, treplacer)

    def test_prob34_ismaximallymixed(self):
        ppauli_from_depol = 3/4
        self.arbpsi = self.pop_state
        onemut = mutation.RandomPauli(self.chromosome_size, self.population_size, ppauli_from_depol)
        
        arbpsi2 = onemut.mutate(self.arbpsi)
        
        np.testing.assert_array_almost_equal(arbpsi2,
                                             np.identity(self.system_dim) / self.system_dim)
                    
    def test_aschannel_isdepol(self):
        # Should be a local depolarizer
        pdepol = 4 / 3  * self.mutation_probability
        tdepol = channel_families.depolarizing(2, pdepol)
        np.testing.assert_array_almost_equal(tdepol, self.rpauli.local_mutation_mat)

    def test_aschannel_purestate(self):
        arbpsi2 = self.rpauli.mutate(self.arbpsi)

        pdepol = 4 / 3  * self.mutation_probability
        tdepol = channel_families.depolarizing(2, pdepol)

        arbpsi2_ref = self.arbpsi.copy()
        for chromosome in range(self.chromosome_size * self.population_size):
            arbpsi2_ref = state_transformations.local_channel(arbpsi2_ref,
                                                              self.system_shape,
                                                              chromosome,
                                                              tdepol)
        np.testing.assert_array_almost_equal(arbpsi2, arbpsi2_ref)

    def test_asrandom_onpurestate(self):
        # The input and output states should have same purity
        arbpsi2 = self.rpauli.mutate(self.arbpsi, random_select=True)
        self.assertAlmostEqual(np.trace(arbpsi2 @ arbpsi2), np.trace(self.arbpsi @ self.arbpsi))

        # in local, should be one of some options for the random select one
        psi_local = state_transformations.partial_trace(self.arbpsi,
                                                        self.system_shape,
                                                        range(1, len(self.system_shape)))
        psi2_local = state_transformations.partial_trace(self.arbpsi,
                                                         self.system_shape,
                                                         range(1, len(self.system_shape)))
        equal_to_some_pauli_rot = False
        for pauli in self.rpauli.unitaries_array:
            if np.allclose(psi2_local, pauli @ psi_local @ pauli.T.conj()):
                equal_to_some_pauli_rot = True
                break
        self.assertTrue(equal_to_some_pauli_rot)

if __name__ == "__main__":

    # Add this test in pyqch
    from pyqch import channel_families
    import numpy as np

    paulis = np.array([np.array([[1,0],[0,1]]),
                       np.array([[0,1],[1,0]]),
                       np.array([[0,-1j],[1j,0]]),
                       np.array([[1,0],[0,-1]])])
    p = 3/4-.0123
    prob_array = np.zeros(paulis.shape[0])
    prob_array[0] = 1 - p
    prob_array[1:] = p / 3

    pdepol = 4 / 3  * p  # 2 ** (2*n) / (2 ** (2*n) - 1) * p

    trandu = channel_families.probabilistic_unitaries(2,
                                                      prob_array,
                                                      paulis)
    tdepol = channel_families.depolarizing(2, pdepol)

    np.testing.assert_array_almost_equal(trandu, tdepol)

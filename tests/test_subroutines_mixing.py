import unittest

import numpy as np
from pyqch import predicates
from pyqch import channel_operations
from pyqch import state_transformations
from pyqch import state_families
from pyqch import channel_families
from pyqch import random_generators
from qga_toolbox.subroutines import mixing

class TestSubroutinesMixing_MixingFixedIndex(unittest.TestCase):

    def setUp(self) -> None:
        self.chromosome_size = 2
        self.population_size = 4

        if self.population_size < 4 or self.population_size % 4 != 0:
            raise ValueError(f'The default population_size has to be at least 4 and divisible by 4 to correctly test MixingFixedIndex, but {self.population_size} was obtained.')

        self.system_shape = (2,) * (self.chromosome_size * self.population_size)
        self.system_dim = 2 ** (self.chromosome_size * self.population_size)
        self.random_state = 24597856786290
        
        self.mixing_index = self.chromosome_size // 2

        self.mixer = mixing.MixingFixedIndex(chromosome_size=self.chromosome_size,
                                             population_size=self.population_size,
                                             mixing_index=self.mixing_index)
        
        self.pop_state = np.zeros((self.system_dim, self.system_dim))
        self.pop_state[0,0] = 1

        self.arbpsi = random_generators.state(self.system_dim, 1, self.random_state)

    def test_shape(self):
        arbpsi2 = self.mixer.mix(self.arbpsi)

        self.assertEqual(arbpsi2.shape, self.arbpsi.shape)
    
    def test_error_if_incorrect_mixing_index(self):
        bad_mixing_index = self.chromosome_size + 1

        with self.assertRaises(ValueError):
            mm = mixing.MixingFixedIndex(chromosome_size = self.chromosome_size,
                                         population_size = self.population_size,
                                         mixing_index = bad_mixing_index)

    def test_mixing_permutation_ok(self):
        self.assertEqual(self.mixer.mixing_permutation[:self.population_size],
                         tuple(range(self.population_size)))
        self.assertEqual(self.mixer.mixing_permutation[self.population_size::2],
                         tuple(range(self.population_size, 2*self.population_size, 2)))
        self.assertEqual(self.mixer.mixing_permutation[self.population_size+1::4],
                         tuple(range(self.population_size+3, 2*self.population_size, 4)))
        self.assertEqual(self.mixer.mixing_permutation[self.population_size+3::4],
                         tuple(range(self.population_size+1, 2*self.population_size, 4)))
        
    def test_commutes_with_pairwise_swaps(self):
        pairwise_swap = tuple([min([i + (-1) ** (i%2), self.population_size-1]) for i in range(self.population_size) ])
        
        preswap = state_transformations.subsystem_permutation(self.arbpsi,
                                                              (2 ** self.chromosome_size,) * self.population_size,
                                                              pairwise_swap)
        mixed_preswap = self.mixer.mix(preswap)

        mixed_noswap = self.mixer.mix(self.arbpsi)
        mixed_postswap = state_transformations.subsystem_permutation(mixed_noswap,
                                                                     (2 ** self.chromosome_size,) * self.population_size,
                                                                     pairwise_swap)
        np.testing.assert_array_almost_equal(mixed_preswap, mixed_postswap)
        
    def test_expected_behaviour_in_computational(self):
        a = 2 ** self.mixing_index
        b = 2 ** (self.chromosome_size - self.mixing_index) - (self.chromosome_size - self.mixing_index > 0)

        site_weight = [2 ** (self.chromosome_size * (self.population_size -site - 1)) for site in range(self.population_size)]
        state_odds =  a + b
        state_evens = 0
        in_index = sum([state_odds * site_weight[site] for site in range(0,self.population_size,2)])
        in_index += sum([state_evens * site_weight[site] for site in range(1,self.population_size,2)])

        mixed_state_odds = a
        mixed_state_evend = b
        ou_index = sum([mixed_state_odds * site_weight[site] for site in range(self.population_size//2, self.population_size, 2)])
        ou_index += sum([mixed_state_evend * site_weight[site] for site in range(self.population_size//2+1, self.population_size, 2)])
        ou_index += sum([state_odds * site_weight[site] for site in range(0, self.population_size//2, 2)])
        ou_index += sum([state_evens * site_weight[site] for site in range(1, self.population_size//2, 2)])

        computational_state = state_families.computational_basis(self.system_dim, in_index)
        computational_state = np.outer(computational_state, computational_state.conj())
        mixed_computation_state = self.mixer.mix(computational_state)
        mixed_computation_state_ref = state_families.computational_basis(self.system_dim, ou_index)
        mixed_computation_state_ref = np.outer(mixed_computation_state_ref, mixed_computation_state_ref.conj())

        np.testing.assert_array_almost_equal(mixed_computation_state, mixed_computation_state_ref)

    def test_uniform_superposition_is_invariant(self):
        in_state = np.full(self.system_dim, 1/np.sqrt(self.system_dim))
        in_state = np.outer(in_state, in_state.conj())

        mixed_state = self.mixer.mix(in_state)

        np.testing.assert_array_almost_equal(in_state, mixed_state)


    def test_mixing_index_chromosome_is_identity(self):
        
        mixer_id = mixing.MixingFixedIndex(chromosome_size=self.chromosome_size,
                                           population_size=self.population_size,
                                           mixing_index=self.chromosome_size)
        
        mixed_arbpsi = mixer_id.mix(self.arbpsi)
        np.testing.assert_array_almost_equal(mixed_arbpsi, self.arbpsi)

    def test_mixing_index_0_size_is_swap(self):
        
        mixer_id = mixing.MixingFixedIndex(chromosome_size=self.chromosome_size,
                                           population_size=self.population_size,
                                           mixing_index=0)
        
        mixed_arbpsi = mixer_id.mix(self.arbpsi)

        swap_lowest_half = tuple([i + (i >= self.population_size//2) * (-1) ** (i % 2) for i in range(self.population_size)])

        mixed_arbpsi_ref = state_transformations.subsystem_permutation(self.arbpsi,
                                                                       (2 ** self.chromosome_size,) * self.population_size,
                                                                       permutation=swap_lowest_half)
        np.testing.assert_array_almost_equal(mixed_arbpsi, mixed_arbpsi_ref)
    
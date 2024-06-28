import unittest

import numpy as np
from pyqch import predicates
from pyqch import channel_operations
from pyqch import state_transformations
from pyqch import channel_families
from pyqch import random_generators
from qga_toolbox.subroutines import mixing

class TestSubroutinesMixing_MixingFixedIndext(unittest.TestCase):

    def setUp(self) -> None:
        self.chromosome_size = 2
        self.population_size = 2

        self.system_shape = (2,) * (self.chromosome_size * self.population_size)
        self.system_dim = 2 ** (self.chromosome_size * self.population_size)
        self.random_state = 24597856786290
        
        mixing_index = self.chromosome_size // 2

        self.mixer = mixing.MixingFixedIndex(chromosome_size=self.chromosome_size,
                                             population_size=self.population_size,
                                             mixing_index=mixing_index)
        
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
                         list(range(self.population_size)))
        self.assertEqual(self.mixer.mixing_permutation[self.population_size::2],
                         list(range(self.population_size, 2*self.population_size, 2)))
        self.assertEqual(self.mixer.mixing_permutation[self.population_size+1::4],
                         list(range(self.population_size+3, 2*self.population_size, 4)))
        self.assertEqual(self.mixer.mixing_permutation[self.population_size+3::4],
                         list(range(self.population_size+1, 2*self.population_size, 4)))
        
    def test_commutes_with_pairwise_swaps(self):
        pass

    def test_expected_behaviour_in_computational(self):
        pass

    def test_expected_behaviour_in_arb_basis(self):
        pass

    def test_mixing_index_0_is_identity(self):
        pass

    def test_mixing_index_chromosome_size_is_swap(self):
        pass
    
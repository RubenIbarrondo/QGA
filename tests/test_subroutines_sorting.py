import unittest

import numpy as np
from pyqch import predicates
from pyqch import channel_operations
from pyqch import state_transformations
from pyqch import channel_families
from pyqch import random_generators
from src.subroutines import sorting

class TestSubroutinesSorting_FullSort(unittest.TestCase):

    def setUp(self) -> None:
        self.chromosome_size = 2
        self.population_size = 2

        self.system_shape = (2,) * (self.chromosome_size * self.population_size)
        self.system_dim = 2 ** (self.chromosome_size * self.population_size)
        self.random_state = 24597856786290
        

        classical_hamiltonian = np.diag(np.arange(2**self.chromosome_size))

        self.sorter = sorting.FullSort(self.chromosome_size, self.population_size, classical_hamiltonian)

        self.pop_state = np.zeros((self.system_dim, self.system_dim))
        self.pop_state[0,0] = 1

        self.arbpsi = random_generators.state(self.system_dim, 1, self.random_state)
    
    def test_shapes(self):

        self.assertEqual(self.sorter.pairwise_sort_mat.shape,
                         (2 ** (2 * 2 * self.chromosome_size), 2 ** (2 * 2 * self.chromosome_size)))
        pop_psi2 = self.sorter.sort(self.pop_state)
        self.assertEqual(pop_psi2.shape, self.pop_state.shape)

        arbpsi = self.sorter.sort(self.arbpsi)
        self.assertEqual(arbpsi.shape, self.arbpsi.shape)
        
    def test_is_channel(self):
        #print(predicates.is_channel(self.sorter.pairwise_sort_mat, show=True))
        self.assertTrue(predicates.is_channel(self.sorter.pairwise_sort_mat))

    def test_gs_is_preserved(self):
        pop_psi2 = self.sorter.sort(self.pop_state)
        np.testing.assert_array_almost_equal(pop_psi2, self.pop_state)

    def test_sorted_is_preserved(self):
        sorted_state = 1
        seq = np.arange(2 ** self.chromosome_size)
        for site in range(self.population_size):
            site_state = np.diag((seq == site) | (site >= seq[-1]))
            sorted_state = np.kron(sorted_state, site_state)

        sorted_state2 = self.sorter.sort(sorted_state)
        np.testing.assert_array_almost_equal(sorted_state2, sorted_state)
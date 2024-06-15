import unittest

import numpy as np
from pyqch import predicates
from pyqch import channel_operations
from pyqch import state_transformations
from pyqch import channel_families
from pyqch import random_generators
from qga_toolbox.subroutines import sorting

class TestSubroutinesSorting_FullSort(unittest.TestCase):

    def setUp(self) -> None:
        self.chromosome_size = 2
        self.population_size = 2

        self.system_shape = (2,) * (self.chromosome_size * self.population_size)
        self.system_dim = 2 ** (self.chromosome_size * self.population_size)
        self.random_state = 24597856786290
        
        # Diagonal Hamiltonian
        classical_hamiltonian = np.diag(np.arange(2**self.chromosome_size))

        self.sorter = sorting.FullSort(self.chromosome_size, self.population_size, classical_hamiltonian)

        self.pop_state = np.zeros((self.system_dim, self.system_dim))
        self.pop_state[0,0] = 1

        self.arbpsi = random_generators.state(self.system_dim, 1, self.random_state)

        # Non-diagonal Hamiltonian
        unit = random_generators.unitary_channel(2**self.chromosome_size, random_state=self.random_state)
        quantum_hamiltonian = (unit @ classical_hamiltonian.reshape((4**self.chromosome_size))).reshape((2**self.chromosome_size, 2**self.chromosome_size))

        self.qsorter = sorting.FullSort(self.chromosome_size, self.population_size, quantum_hamiltonian)

        # Getting the ground state
        qpsi_gs = np.outer(self.qsorter.basis[0,:], self.qsorter.basis[0,:].conj())

        self.qpop_state = 1
        for site in range(self.population_size):
            self.qpop_state = np.kron(qpsi_gs, self.qpop_state)
    
    def _get_sorted_state(self, sorter):
        sorted_state = 1
        seq = np.arange(2 ** self.chromosome_size)
        for site in range(self.population_size):
            if site >= seq[-1]:
                index = seq[-1]
            else:
                index = site
            site_state = np.outer(sorter.basis[index,:], sorter.basis[index,:].conj())
            sorted_state = np.kron(sorted_state, site_state)
        return sorted_state

    def _get_unsorted_state(self, sorter):
        unsorted_state = 1
        seq = np.arange(2 ** self.chromosome_size)
        for site in range(self.population_size):
            if site >= seq[-1]:
                index = seq[-1]
            else:
                index = site
            site_state = np.outer(sorter.basis[index,:], sorter.basis[index,:].conj())
            unsorted_state = np.kron(site_state, unsorted_state)
        return unsorted_state

    def test_shapes(self):

        self.assertEqual(self.sorter.pairwise_sort_mat.shape,
                         (2 ** (2 * 2 * self.chromosome_size), 2 ** (2 * 2 * self.chromosome_size)))
        pop_psi2 = self.sorter.sort(self.pop_state)
        self.assertEqual(pop_psi2.shape, self.pop_state.shape)

        arbpsi = self.sorter.sort(self.arbpsi)
        self.assertEqual(arbpsi.shape, self.arbpsi.shape)
        
    def test_is_channel(self):
        self.assertTrue(predicates.is_channel(self.sorter.pairwise_sort_mat))

    def test_gs_is_preserved(self):
        pop_psi2 = self.sorter.sort(self.pop_state)
        np.testing.assert_array_almost_equal(pop_psi2, self.pop_state)

    def test_it_sorts(self):
        sorted_state = self._get_sorted_state(self.sorter)
        unsorted_state = self._get_unsorted_state(self.sorter)
        unsorted_state2 = self.sorter.sort(unsorted_state)
        np.testing.assert_array_almost_equal(unsorted_state2, sorted_state)

    def test_sorted_is_preserved(self):
        sorted_state = self._get_sorted_state(self.sorter)

        sorted_state2 = self.sorter.sort(sorted_state)
        np.testing.assert_array_almost_equal(sorted_state2, sorted_state)

    def test_is_channel_notcanonical(self):
        self.assertTrue(predicates.is_channel(self.qsorter.pairwise_sort_mat))

    def test_gs_is_preserved_notcanonical(self):
        pop_state2 = self.qsorter.sort(self.qpop_state)
        np.testing.assert_array_almost_equal(pop_state2, self.qpop_state)

    def test_sorted_is_preserved_notcanonical(self):
        sorted_state = self._get_sorted_state(self.qsorter)
        sorted_state2 = self.qsorter.sort(sorted_state)
        np.testing.assert_array_almost_equal(sorted_state2, sorted_state)
    
    def test_it_sorts_notcanonical(self):
        sorted_state = self._get_sorted_state(self.qsorter)
        unsorted_state = self._get_unsorted_state(self.qsorter)
        unsorted_state2 = self.qsorter.sort(unsorted_state)
        np.testing.assert_array_almost_equal(unsorted_state2, sorted_state)

        
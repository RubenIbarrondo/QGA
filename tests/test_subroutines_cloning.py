import unittest

import numpy as np
from pyqch import predicates
from pyqch import channel_operations
from pyqch import state_transformations
from pyqch import state_families
from qga_toolbox.subroutines import cloning

class TestSubroutinesCloning_UQCM(unittest.TestCase):

    def setUp(self) -> None:
        self.chromosome_size = 2
        self.population_size = 4

        self.uqcm = cloning.UQCM(self.chromosome_size, self.population_size)

        self.psi = np.zeros((self.uqcm.dim, self.uqcm.dim))
        self.psi[0,0] = 1

        self.pop_state = np.zeros((self.uqcm.dim ** (self.population_size // 2),) * 2)
        self.pop_state[0,0] = 1


    def test_shapes(self):
        self.assertEqual(self.uqcm.clone_mat.shape, (self.uqcm.dim ** 4, self.uqcm.dim ** 2))
            
    def test_is_channel(self):
        self.assertTrue(predicates.is_channel(self.uqcm.clone_mat))

    def test_symmetry(self):
        choi = channel_operations.choi_state(self.uqcm.clone_mat)
        choi_swap = state_transformations.subsystem_permutation(choi,
                                                                (self.uqcm.dim,) * 3,
                                                                (1, 0, 2))
        np.testing.assert_array_almost_equal(choi, choi_swap)

    def test_cloning_pure_states(self):
        psi_clone = self.uqcm.clone(self.psi, single_individual=True)

        rho1 = state_transformations.partial_trace(psi_clone,
                                                   (self.uqcm.dim,)*2,
                                                   1)
        rho2 = state_transformations.partial_trace(psi_clone,
                                                   (self.uqcm.dim,)*2,
                                                   0)
        
        # Re-checking symmetry
        np.testing.assert_array_almost_equal(rho1, rho2)

        # Comparing with the theoretical prediction
        d = self.uqcm.dim
        rho1_ref = .5 / (d + 1) * ((d+2) * self.psi + np.identity(d))
        np.testing.assert_array_almost_equal(rho1_ref, rho1)
    
    def test_maximally_mixed(self):
        d = self.uqcm.dim
        mixed = np.identity(d)
        mixed_clone = self.uqcm.clone(mixed, single_individual=True)

        swap = np.einsum("ijkl->jikl", np.identity(d ** 2).reshape((d,)*4)).reshape((d**2, d**2))
        sym_proj = (np.identity(d ** 2) + swap) / 2
        mixed_clone_ref = 2 / (d + 1) * sym_proj

        np.testing.assert_array_almost_equal(mixed_clone, mixed_clone_ref)

    def test_poplevel(self):

        pop_state_clone = self.uqcm.clone(self.pop_state)

        self.assertEqual(pop_state_clone.shape, (self.uqcm.dim ** self.population_size, self.uqcm.dim ** self.population_size))

        self.assertTrue(predicates.is_density_matrix(pop_state_clone))

        # Testing symmetry
        pop_state_clone_swap = state_transformations.subsystem_permutation(pop_state_clone,
                                                                           (self.uqcm.dim ** (self.population_size // 2),) * 2,
                                                                           (1,0))
        
        np.testing.assert_array_almost_equal(pop_state_clone, pop_state_clone_swap)

        # Testing the reduced state of the first
        rho1 = state_transformations.partial_trace(pop_state_clone,
                                                    (self.uqcm.dim,) * self.population_size,
                                                    list(range(1,self.population_size)))
        d = self.uqcm.dim
        rho1_ref = .5 / (d + 1) * ((d+2) * self.psi + np.identity(d))

        np.testing.assert_array_almost_equal(rho1, rho1_ref)
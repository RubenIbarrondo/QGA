import unittest

import numpy as np
from pyqch import predicates
from pyqch import channel_operations
from pyqch import state_transformations
from pyqch import random_generators
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
    

    def test_error_if_state_too_big(self):
        too_big_pop_state = np.zeros((self.uqcm.dim ** (self.population_size),) * 2)
        too_big_pop_state[0,0] = 1

        with self.assertRaises(ValueError):
            self.uqcm.clone(too_big_pop_state)


class TestSubroutinesCloning_BCQO(unittest.TestCase):

    def setUp(self) -> None:
        self.chromosome_size = 2
        self.population_size = 4
        self.cloning_basis = 1

        self.bcqo = cloning.BCQO(self.chromosome_size, self.population_size, self.cloning_basis)

        self.psi = random_generators.state(self.bcqo.dim, rank=1, random_state=123)

        self.pop_state = np.zeros((self.bcqo.dim ** (self.population_size // 2),) * 2)
        self.pop_state[0,0] = 1


    def test_shapes(self):
        self.assertEqual(self.bcqo.clone_mat.shape, (self.bcqo.dim ** 4, self.bcqo.dim ** 2))
            
    def test_is_channel(self):
        self.assertTrue(predicates.is_channel(self.bcqo.clone_mat))

    def test_symmetry(self):
        choi = channel_operations.choi_state(self.bcqo.clone_mat)
        choi_swap = state_transformations.subsystem_permutation(choi,
                                                                (self.bcqo.dim,) * 3,
                                                                (1, 0, 2))
        np.testing.assert_array_almost_equal(choi, choi_swap)

    def test_cloning_pure_states(self):
        psi_clone = self.bcqo.clone(self.psi, single_individual=True)

        psi_ref = np.zeros((self.bcqo.dim,)*4, dtype=complex)
        for i in range(self.bcqo.dim):
            for j in range(self.bcqo.dim):
                psi_ref[i,i,j,j] = self.psi[i,j]
        psi_ref = psi_ref.reshape(((self.bcqo.dim**2,)*2))

        np.testing.assert_array_almost_equal(psi_clone, psi_ref)
    
    def test_cloning_pure_states_random_basis(self):
        from scipy.stats import unitary_group

        u = unitary_group.rvs(2 ** self.chromosome_size, random_state=np.random.default_rng(123))
        bcqo_u = cloning.BCQO(self.chromosome_size, self.population_size,
                              cloning_basis=u)
        
        psi_clone = bcqo_u.clone(self.psi, single_individual=True)

        psi_u = u @ self.psi @ u.T.conj()
        psi_ref = np.zeros((self.bcqo.dim,)*4, dtype=complex)
        for i in range(bcqo_u.dim):
            for j in range(bcqo_u.dim):
                psi_ref[i,i,j,j] = psi_u[i,j]
        psi_ref = np.kron(u.T.conj(), u.T.conj()) @ psi_ref.reshape(((bcqo_u.dim**2,)*2)) @ np.kron(u, u)

        np.testing.assert_array_almost_equal(psi_clone, psi_ref)
    
    def test_maximally_mixed(self):
        d = self.bcqo.dim
        mixed = np.identity(d)
        mixed_clone = self.bcqo.clone(mixed, single_individual=True)

        mixed_ref = np.zeros((self.bcqo.dim,)*4, dtype=complex)
        for i in range(self.bcqo.dim):
            for j in range(self.bcqo.dim):
                mixed_ref[i,i,j,j] = mixed[i,j]
        mixed_ref = mixed_ref.reshape((self.bcqo.dim**2,)*2)

        np.testing.assert_array_almost_equal(mixed_clone, mixed_ref)

    def test_poplevel(self):

        pop_state_clone = self.bcqo.clone(self.pop_state)


        self.assertEqual(pop_state_clone.shape, (self.bcqo.dim ** self.population_size, self.bcqo.dim ** self.population_size))

        self.assertTrue(predicates.is_density_matrix(pop_state_clone))

        # Testing symmetry
        pop_state_clone_swap = state_transformations.subsystem_permutation(pop_state_clone,
                                                                           (self.bcqo.dim ** (self.population_size // 2),) * 2,
                                                                           (1,0))
        
        np.testing.assert_array_almost_equal(pop_state_clone, pop_state_clone_swap)

        # Testing that the reduced state of the first didn't change
        rho1 = state_transformations.partial_trace(pop_state_clone,
                                                    (self.bcqo.dim,) * self.population_size,
                                                    list(range(1,self.population_size)))
        rho1_ref = state_transformations.partial_trace(self.pop_state,
                                                    (self.bcqo.dim,) * (self.population_size//2),
                                                    list(range(1,self.population_size//2)))
        np.testing.assert_array_almost_equal(rho1, rho1_ref)
    

    def test_error_if_state_too_big(self):
        too_big_pop_state = np.zeros((self.bcqo.dim ** (self.population_size),) * 2)
        too_big_pop_state[0,0] = 1

        with self.assertRaises(ValueError):
            self.bcqo.clone(too_big_pop_state)
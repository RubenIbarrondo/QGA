import unittest
import numpy as np

from qga_toolbox import feature_trackers
from pyqch import random_generators

class TestFeatureTrackers_IndividualEigenstateFidelity(unittest.TestCase):

    def test_shape(self):
        cs = 2
        ps = 4
        eb = np.identity(2 ** cs)

        state = np.zeros((2**(cs*ps), 2**(cs*ps)))
        state[0,0] = 1

        ft = feature_trackers.IndividualEigenstateFidelity(chromosome_size=cs, population_size=ps, energy_basis=eb)

        fmat = ft.track(state)
        self.assertEqual(fmat.shape, (ps, eb.shape[0]))

    def test_canonical_binary(self):
        cs = 2
        ps = 4
        eb = np.identity(2 ** cs)

        state = np.zeros((2**(cs*ps), 2**(cs*ps)))
        state[0,0] = 1

        ft = feature_trackers.IndividualEigenstateFidelity(chromosome_size=cs, population_size=ps, energy_basis=eb)

        fmat = ft.track(state)

        fmat_ref = np.zeros((ps, eb.shape[0]))
        fmat_ref[:,0] = 1

        np.testing.assert_array_almost_equal(fmat, fmat_ref)

    def test_canonical_generic(self):
        cs = 2
        ps = 4
        eb = np.identity(2 ** cs)

        weights = np.array([np.linspace(reg, reg+1, eb.shape[0]) / np.sum(np.linspace(reg, reg+1, eb.shape[0])) for reg in range(ps)])

        state = 1
        for reg in range(ps):
            state = np.kron(state, np.diag(weights[reg]))

        ft = feature_trackers.IndividualEigenstateFidelity(chromosome_size=cs, population_size=ps, energy_basis=eb)

        fmat = ft.track(state)
        fmat_ref = weights
        np.testing.assert_array_almost_equal(fmat, fmat_ref)

    def test_probability_distributions(self):
        cs = 2
        ps = 4
        eb = np.identity(2 ** cs)
        ft = feature_trackers.IndividualEigenstateFidelity(chromosome_size=cs, population_size=ps, energy_basis=eb)

        state = random_generators.state(2**(cs*ps), rank=1)

        fmat = ft.track(state)
        atol = 1e-6
        np.testing.assert_array_less(-atol, fmat)

        np.testing.assert_array_almost_equal(np.sum(fmat, axis=1), 1)

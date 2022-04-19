import os
import numpy as np
import quantum_mats as qm
from copy import deepcopy
from time import time
from scipy.stats import unitary_group
from scipy import linalg

import QGA_BCQO_sim as bqga
import QGA_UQCM_sim as uqga


def criteria(x, y):
    return sum(int(xi) * 2 ** (len(x) - i - 1) for i, xi in enumerate(x)) > sum(
        int(yi) * 2 ** (len(y) - i - 1) for i, yi in enumerate(y))


def pm_sweep_from_previous(pm_rate_min, pm_rate_max, pm_number, number_of_problems):
    """The idea is to try the SAME Hp's and Initial Pops but performing a value
    sweep in the mutation probability pm."""
    c = 2
    n = 4
    pm0 = [1 / n / c / 3 / 3] * 3
    pm_values = [[pm]*3 for pm in np.linspace(pm_rate_min * pm0[0], pm_rate_max * pm0[0], pm_number)]
    mu = [np.array([[0, 1], [1, 0]]),
          np.array([[0, -1j], [1j, 0]]),
          np.array([[1, 0], [0, -1]])]

    number_of_initial_populations = 10
    number_of_generations = 10

    big_dirs = ("out_BCQO_nm", "out_UQCM_nm", "out_inits", "out_Ups")
    for dirpath in big_dirs:
        if not os.path.exists(dirpath):
            raise Exception(dirpath + " does not exist.")

    t1 = time()
    for i in range(number_of_problems):
        pathname = "problem_%d" % i

        if os.path.exists(big_dirs[-1] + "/" + pathname + ".npy"):
            up = np.load(big_dirs[-1] + "/" + pathname + ".npy")
        else:
            print(big_dirs[-1] + "/" + pathname, "does not exist, the index will be skipped.")
            continue

        # Create output directories
        for j in range(number_of_initial_populations):
            for pm_ind in range(pm_number):
                if not os.path.exists("out_BCQO_pms" + "/" + pathname + '/pms_{:03d}'.format(pm_ind)):
                    os.makedirs("out_BCQO_pms" + "/" + pathname + '/pms_{:03d}'.format(pm_ind))
                if not os.path.exists("out_UQCM_pms" + "/" + pathname + '/pms_{:03d}'.format(pm_ind)):
                    os.makedirs("out_UQCM_pms" + "/" + pathname + '/pms_{:03d}'.format(pm_ind))

        for j in range(number_of_initial_populations):
            rho_population_vec = np.load(big_dirs[2] + "/" + pathname + '/initial_population_{:03d}'.format(j) + ".npy")
            rho_population_vec = rho_population_vec.T
            rho_population_mat = np.kron(rho_population_vec.reshape((rho_population_vec.shape[1], 1)), rho_population_vec.conjugate())
            rho_population_mat = rho_population_mat / np.trace(rho_population_mat)
            rho_population = qm.rho.gen_rho_from_matrix(rho_population_mat)

            for pm_ind, pm in enumerate(pm_values):
                # BCQO pms
                rho_final, ft = bqga.quantum_genetic_algorithm(criteria, fitness_basis=up,
                                                               init_population=rho_population, n=n, cl=c,
                                                               generation_number=number_of_generations,
                                                               pm=pm, mutation_unitary=mu,
                                                               projection_method="ptrace", store_path=None,
                                                               track_fidelity=[up[:, i] for i in range(up.shape[0])])
                # Save...
                np.save("out_BCQO_pms" + "/" + pathname + '/pms_{:03d}'.format(pm_ind) + '/fidelity_tracks_{:03d}'.format(j), np.array(ft, dtype=np.complex64))
                np.save("out_BCQO_pms" + "/" + pathname + '/pms_{:03d}'.format(pm_ind) + '/final_population_{:03d}'.format(j), np.array(rho_final.partial_trace(list(range(n * c))).get_matrix(), dtype=np.complex64))

                # UQCM pms
                rho_final, ft = uqga.quantum_genetic_algorithm(criteria, fitness_basis=up,
                                                               init_population=rho_population, n=n, cl=c,
                                                               generation_number=number_of_generations,
                                                               pm=pm, mutation_unitary=mu,
                                                               projection_method="ptrace", store_path=None,
                                                               track_fidelity=[up[:, i] for i in range(up.shape[0])])
                # Save...
                np.save("out_UQCM_pms" + "/" + pathname + '/pms_{:03d}'.format(pm_ind) + '/fidelity_tracks_{:03d}'.format(j), np.array(ft, dtype=np.complex64))
                np.save("out_UQCM_pms" + "/" + pathname + '/pms_{:03d}'.format(pm_ind) + '/final_population_{:03d}'.format(j), np.array(rho_final.partial_trace(list(range(n * c))).get_matrix(), dtype=np.complex64))
            print(pathname, "time: ", time() - t1)


if __name__ == '__main__':
    pm_sweep_from_previous(pm_rate_min=.1, pm_rate_max=3.0, pm_number=10, number_of_problems=25)

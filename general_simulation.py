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


if __name__ == '__main__':
    c = 2
    n = 4
    pm = [1 / n / c / 3 / 3] * 3
    mu = [np.array([[0, 1], [1, 0]]),
          np.array([[0, -1j], [1j, 0]]),
          np.array([[1, 0], [0, -1]])]

    number_of_Hps = 200
    number_of_initial_populations = 10
    number_of_generations = {"BCQO_nm": 5,
                             "BCQO_wm": 10,
                             "UQCM_nm": 5,
                             "UQCM_wm": 10}

    big_dirs = ("out_BCQO_nm", "out_BCQO_wm", "out_UQCM_nm", "out_UQCM_wm", "out_inits", "out_Ups")

    for dirpath in big_dirs:
        if not os.path.exists(dirpath):
            os.makedirs(dirpath)

    t1 = time()
    for i in range(number_of_Hps):
        pathname = "problem_%d" % i
        up = unitary_group.rvs(2 ** c)

        if not os.path.exists(big_dirs[-1] + "/" + pathname + ".npy"):

            np.save(big_dirs[-1] + "/" + pathname, np.array(up, dtype=np.complex64))

            for dirpath in big_dirs[:-1]:
                if not os.path.exists(dirpath + "/" + pathname):
                    os.makedirs(dirpath + "/" + pathname)
                else:
                    raise Exception(dirpath + "/" + pathname + " already exists.")
        else:
            print(big_dirs[-1] + "/" + pathname, "already exists, the index will be skipped.")
            continue

        for j in range(number_of_initial_populations):

            rho_population = qm.rho.gen_random_rho(n * c)

            w, v = linalg.eigh(rho_population.get_matrix(), eigvals=(2 ** (n * c)-1, 2 ** (n * c)-1))
            np.save(big_dirs[4] + "/" + pathname + '/initial_population_{:03d}'.format(j), np.array(v, dtype=np.complex64))

            # BCQO nm
            rho_final, ft = bqga.quantum_genetic_algorithm(criteria, fitness_basis=up,
                                                           init_population=rho_population, n=n, cl=c,
                                                           generation_number=number_of_generations["BCQO_nm"],
                                                           pm=0, mutation_unitary="I",
                                                           projection_method="ptrace", store_path=None,
                                                           track_fidelity=[up[:, i] for i in range(up.shape[0])])
            # Save...
            np.save(big_dirs[0] + "/" + pathname + '/fidelity_tracks_{:03d}'.format(j), np.array(ft, dtype=np.complex64))
            np.save(big_dirs[0] + "/" + pathname + '/final_population_{:03d}'.format(j), np.array(rho_final.partial_trace(list(range(n * c))).get_matrix(), dtype=np.complex64))

            # BCQO wm
            rho_final, ft = bqga.quantum_genetic_algorithm(criteria, fitness_basis=up,
                                                           init_population=rho_population, n=n, cl=c,
                                                           generation_number=number_of_generations["BCQO_wm"],
                                                           pm=pm, mutation_unitary=mu,
                                                           projection_method="ptrace", store_path=None,
                                                           track_fidelity=[up[:, i] for i in range(up.shape[0])])
            # Save...
            np.save(big_dirs[1] + "/" + pathname + '/fidelity_tracks_{:03d}'.format(j), np.array(ft, dtype=np.complex64))
            np.save(big_dirs[1] + "/" + pathname + '/final_population_{:03d}'.format(j), np.array(rho_final.partial_trace(list(range(n * c))).get_matrix(), dtype=np.complex64))

            # UQCM nm
            rho_final, ft = uqga.quantum_genetic_algorithm(criteria, fitness_basis=up,
                                                           init_population=rho_population, n=n, cl=c,
                                                           generation_number=number_of_generations["UQCM_nm"],
                                                           pm=0, mutation_unitary="I",
                                                           projection_method="ptrace", store_path=None,
                                                           track_fidelity=[up[:, i] for i in range(up.shape[0])])
            # Save...
            np.save(big_dirs[2] + "/" + pathname + '/fidelity_tracks_{:03d}'.format(j), np.array(ft, dtype=np.complex64))
            np.save(big_dirs[2] + "/" + pathname + '/final_population_{:03d}'.format(j), np.array(rho_final.partial_trace(list(range(n * c))).get_matrix(), dtype=np.complex64))

            # UQCM wm
            rho_final, ft = uqga.quantum_genetic_algorithm(criteria, fitness_basis=up,
                                                           init_population=rho_population, n=n, cl=c,
                                                           generation_number=number_of_generations["UQCM_wm"],
                                                           pm=pm, mutation_unitary=mu,
                                                           projection_method="ptrace", store_path=None,
                                                           track_fidelity=[up[:, i] for i in range(up.shape[0])])
            # Save...
            np.save(big_dirs[3] + "/" + pathname + '/fidelity_tracks_{:03d}'.format(j), np.array(ft, dtype=np.complex64))
            np.save(big_dirs[3] + "/" + pathname + '/final_population_{:03d}'.format(j), np.array(rho_final.partial_trace(list(range(n * c))).get_matrix(), dtype=np.complex64))
        print(pathname, "time: ", time() - t1)

import os
import numpy as np
import quantum_mats as qm
from copy import deepcopy
from time import time
from scipy.stats import unitary_group
from scipy import linalg

from CGA import *


def cGA_simulation(number_of_first_Hp, number_of_last_Hp):
    big_dirs = ("out_cGAai", "out_cGAaii", "out_cGAbi", "out_cGAbii")
    number_of_initial_populations = 100
    generations = 10
    pm = 0.125

    for dirpath in big_dirs:
        if not os.path.exists(dirpath):
            os.mkdir(dirpath)

    for i in range(number_of_first_Hp, number_of_last_Hp):
        pathname = "problem_%d" % i

        if os.path.exists("out_Ups/" + pathname + ".npy"):
            up = np.load("out_Ups/" + pathname + ".npy")
            H = up @ np.diag([1, 2, 3, 4]) @ up.transpose().conjugate()
        for dirpath in big_dirs:
            if not os.path.exists(dirpath + "/" + pathname):
                os.mkdir(dirpath + "/" + pathname)

        for init_pop_index in range(number_of_initial_populations):
            # Initial population
            # WARNING! Initial populations are different!!!
            # rho_population_mat = np.load("out_inits/problem_{:d}/initial_population_{:03d}".format(i,j) + ".npy")
            # rho_population = qm.rho.gen_rho_from_matrix(rho_population_mat)
            init_pop = [qm.rho.gen_random_rho(2, asvector=True) for i in range(4)]

            # Computing the results
            popai, ftai = CGA_1(H, init_pop, 4, 2, generations, pm, crossover_a, mutation_i)
            popbi, ftbi = CGA_1(H, init_pop, 4, 2, generations, pm, crossover_b, mutation_i)
            popaii, ftaii = CGA_1(H, init_pop, 4, 2, generations, pm, crossover_a, mutation_ii)
            popbii, ftbii = CGA_1(H, init_pop, 4, 2, generations, pm, crossover_b, mutation_ii)

            # Saving results
            # Fidelities
            np.save(big_dirs[0] + '/' + pathname + '/fidelity_tracks_{:03d}'.format(init_pop_index),
                    np.array(ftai, dtype=np.complex64))
            np.save(big_dirs[1] + '/' + pathname + '/fidelity_tracks_{:03d}'.format(init_pop_index),
                    np.array(ftaii, dtype=np.complex64))
            np.save(big_dirs[2] + '/' + pathname + '/fidelity_tracks_{:03d}'.format(init_pop_index),
                    np.array(ftbi, dtype=np.complex64))
            np.save(big_dirs[3] + '/' + pathname + '/fidelity_tracks_{:03d}'.format(init_pop_index),
                    np.array(ftbii, dtype=np.complex64))

            # Final pop
            np.save(big_dirs[0] + '/' + pathname + '/final_pop_{:03d}'.format(init_pop_index),
                    np.array(popai, dtype=np.complex64))
            np.save(big_dirs[1] + '/' + pathname + '/final_pop_{:03d}'.format(init_pop_index),
                    np.array(popaii, dtype=np.complex64))
            np.save(big_dirs[2] + '/' + pathname + '/final_pop_{:03d}'.format(init_pop_index),
                    np.array(popbi, dtype=np.complex64))
            np.save(big_dirs[3] + '/' + pathname + '/final_pop_{:03d}'.format(init_pop_index),
                    np.array(popbii, dtype=np.complex64))


if __name__ == '__main__':
    t1 = time()
    cGA_simulation(number_of_first_Hp=0, number_of_last_Hp=200)
    print(time() - t1)

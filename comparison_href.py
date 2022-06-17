"""
.. module:: comparison_href.py
    :synopsis: Performs a comparison between the four QGA variants and some classical variants.
.. moduleauthor::  Ruben Ibarrondo (rubenibarrondo@gmail.com)
"""
import time

import numpy as np
import QGA_UQCM_sim as uqcm
import QGA_BCQO_sim as bcqo
from CGA import *
import quantum_mats as qm
import os
import time


def ref_comparison(H, number_of_initial_populations, generations, save_path=None, show_figs=False):
    if show_figs:
        import matplotlib.pyplot as plt
        fig, axs = plt.subplots(2, 2, figsize=[15, 6])
        axs[0, 0].set_title(save_path)
        axs[1, 0].set_title(save_path)

    if save_path is not None:
        try:
            os.mkdir(save_path)
            for dirname in ["cGA1_inits", "cGA2_inits", "QGA_inits"]:
                os.mkdir(save_path+'/'+dirname)
            for dirname in ["cGA2", "cGAai", "cGAbi", "cGAaii", "cGAbii"]:
                os.mkdir(save_path+'/'+dirname)
            for dirname in ["QGAunm", "QGAuwm", "QGAbnm", "QGAbwm"]:
                os.mkdir(save_path+'/'+dirname)
        except FileExistsError as fer:
            print("Ignored: ", str(fer))

    # Define parameters
    n = 4
    cl = 2
    pm = 1 / n / cl
    criteria = lambda x, y: sum(int(xi) * 2 ** (len(x) - i - 1) for i, xi in enumerate(x)) > sum(
        int(yi) * 2 ** (len(y) - i - 1) for i, yi in enumerate(y))
    ppu = "I"
    mu = [np.array([[0, 1], [1, 0]]),
          np.array([[0, -1j], [1j, 0]]),
          np.array([[1, 0], [0, -1]])]
    pm_arr = [1 / n / cl / 3] * 3

    first_init_pop_indiex = 0
    while os.path.exists(save_path + '/cGA1_inits/initial_pop_{:03d}'.format(first_init_pop_indiex)):
        first_init_pop_indiex += 1

    for init_pop_index in range(first_init_pop_indiex, first_init_pop_indiex + number_of_initial_populations):
        # cGA's
        init_pop = [qm.rho.gen_random_rho(2, asvector=True) for i in range(4)]
        bin_init_pop = [1 * (state == np.max(state)) for state in init_pop]

        popai, ftai = CGA_1(H, init_pop, 4, 2, generations, pm, crossover_a, mutation_i)
        popbi, ftbi = CGA_1(H, init_pop, 4, 2, generations, pm, crossover_b, mutation_i)
        popaii, ftaii = CGA_1(H, init_pop, 4, 2, generations, pm, crossover_a, mutation_ii)
        popbii, ftbii = CGA_1(H, init_pop, 4, 2, generations, pm, crossover_b, mutation_ii)
        if np.linalg.norm(H-np.diag(np.diag(H))) < 1e-5:
            pop2, ft2 = CGA_2(H, bin_init_pop, 4, 2, generations, pm, None, None)
        else:
            pop2 = "Invalid Hamiltonian."
            ft2 = "Invalid Hamiltonian."

        if save_path is not None:
            # Initial pop
            np.save(save_path + '/cGA1_inits/initial_pop_{:03d}'.format(init_pop_index),
                    np.array(init_pop, dtype=np.complex64))
            np.save(save_path + '/cGA2_inits/initial_pop_{:03d}'.format(init_pop_index),
                    np.array(bin_init_pop, dtype=np.complex64))

            # Fidelities
            np.save(save_path + '/cGAai/fidelity_tracks_{:03d}'.format(init_pop_index),
                    np.array(ftai, dtype=np.complex64))
            np.save(save_path + '/cGAbi/fidelity_tracks_{:03d}'.format(init_pop_index),
                    np.array(ftbi, dtype=np.complex64))
            np.save(save_path + '/cGAaii/fidelity_tracks_{:03d}'.format(init_pop_index),
                    np.array(ftaii, dtype=np.complex64))
            np.save(save_path + '/cGAbii/fidelity_tracks_{:03d}'.format(init_pop_index),
                    np.array(ftbii, dtype=np.complex64))
            np.save(save_path + '/cGA2/fidelity_tracks_{:03d}'.format(init_pop_index),
                    np.array(ft2))  # This will never be complex

            # Final pop
            np.save(save_path + '/cGAai/final_pop_{:03d}'.format(init_pop_index),
                    np.array(popai, dtype=np.complex64))
            np.save(save_path + '/cGAbi/final_pop_{:03d}'.format(init_pop_index),
                    np.array(popbi, dtype=np.complex64))
            np.save(save_path + '/cGAaii/final_pop_{:03d}'.format(init_pop_index),
                    np.array(popaii, dtype=np.complex64))
            np.save(save_path + '/cGAbii/final_pop_{:03d}'.format(init_pop_index),
                    np.array(popbii, dtype=np.complex64))
            np.save(save_path + '/cGA2/final_pop_{:03d}'.format(init_pop_index),
                    np.array(pop2))  # This will never be complex

        if show_figs:
            axs[0, 0].plot(ftai[:, 0, 0], "C0", label='1.a-i')
            axs[0, 0].plot(ftbi[:, 0, 0], "C1", label='1.b-i')
            axs[0, 0].plot(ftaii[:, 0, 0], "C2", label='1.a-ii')
            axs[0, 0].plot(ftbii[:, 0, 0], "C3", label='1.b-ii')
            if type(ft2) != str:
                axs[0, 0].plot(ft2[:, 0, 0], "C4", label='2.')

        # QGA's
        rho_vector = init_pop[0]
        for state in init_pop[1:]:
            rho_vector = np.kron(rho_vector, state)
        rho_vector = rho_vector / np.linalg.norm(rho_vector)
        mat_form = np.kron(rho_vector.reshape((rho_vector.shape[0], 1)), rho_vector.conjugate())
        rho_population = qm.rho.gen_rho_from_matrix(mat_form)

        energies, eigenstates = get_eigenbasis(H)
        uu = np.zeros((4, 4), dtype=complex)
        for i, state in enumerate(eigenstates):
            uu[:, i] = state
        tf = eigenstates

        rho_finalunm, ftunm = uqcm.quantum_genetic_algorithm(criteria, fitness_basis=uu,
                                                          init_population=rho_population, n=n, cl=cl,
                                                          generation_number=generations, pm=0, mutation_unitary='I',
                                                          projection_method="ptrace", pre_projection_unitary=ppu,
                                                          store_path=None,
                                                          track_fidelity=tf)

        rho_finaluwm, ftuwm = uqcm.quantum_genetic_algorithm(criteria, fitness_basis=uu,
                                                          init_population=rho_population, n=n, cl=cl,
                                                          generation_number=generations, pm=pm_arr, mutation_unitary=mu,
                                                          projection_method="ptrace", pre_projection_unitary=ppu,
                                                          store_path=None,
                                                          track_fidelity=tf)

        rho_finalbnm, ftbnm = bcqo.quantum_genetic_algorithm(criteria, fitness_basis=uu,
                                                          init_population=rho_population, n=n, cl=cl,
                                                          generation_number=generations, pm=0, mutation_unitary='I',
                                                          projection_method="ptrace", pre_projection_unitary=ppu,
                                                          store_path=None,
                                                          track_fidelity=tf)

        rho_finalbwm, ftbwm = bcqo.quantum_genetic_algorithm(criteria, fitness_basis=uu,
                                                          init_population=rho_population, n=n, cl=cl,
                                                          generation_number=generations, pm=pm_arr, mutation_unitary=mu,
                                                          projection_method="ptrace", pre_projection_unitary=ppu,
                                                          store_path=None,
                                                          track_fidelity=tf)

        if save_path is not None:
            # Initial pop
            w, v = np.linalg.eigh(rho_population.get_matrix())
            np.save(save_path + "/QGA_inits" + '/initial_population_{:03d}'.format(init_pop_index),
                    np.array(v, dtype=np.complex64))

            # Fidelities
            np.save(save_path + "/QGAunm" + '/fidelity_tracks_{:03d}'.format(init_pop_index),
                    np.array(ftunm, dtype=np.complex64))
            np.save(save_path + "/QGAbnm" + '/fidelity_tracks_{:03d}'.format(init_pop_index),
                    np.array(ftbnm, dtype=np.complex64))
            np.save(save_path + "/QGAuwm" + '/fidelity_tracks_{:03d}'.format(init_pop_index),
                    np.array(ftuwm, dtype=np.complex64))
            np.save(save_path + "/QGAbwm" + '/fidelity_tracks_{:03d}'.format(init_pop_index),
                    np.array(ftbwm, dtype=np.complex64))

            # Final state
            np.save(save_path + "/QGAunm" + '/final_population_{:03d}'.format(init_pop_index),
                    np.array(rho_finalunm.partial_trace(list(range(n * cl))).get_matrix(), dtype=np.complex64))
            np.save(save_path + "/QGAbnm" + '/final_population_{:03d}'.format(init_pop_index),
                    np.array(rho_finalbnm.partial_trace(list(range(n * cl))).get_matrix(), dtype=np.complex64))
            np.save(save_path + "/QGAuwm" + '/final_population_{:03d}'.format(init_pop_index),
                    np.array(rho_finaluwm.partial_trace(list(range(n * cl))).get_matrix(), dtype=np.complex64))
            np.save(save_path + "/QGAbwm" + '/final_population_{:03d}'.format(init_pop_index),
                    np.array(rho_finalbwm.partial_trace(list(range(n * cl))).get_matrix(), dtype=np.complex64))

        if show_figs:
            axs[1, 0].plot(ftunm[:, 1, 0, 0], "C0", label='uqcm-nm')
            axs[1, 0].plot(ftbnm[:, 1, 0, 0], "C1", label='bcqo-nm')
            axs[1, 0].plot(ftuwm[:, 1, 0, 0], "C2", label='uqcm-wm')
            axs[1, 0].plot(ftbwm[:, 1, 0, 0], "C3", label='bcqo-wm')

    if show_figs:
        axs[0, 0].legend()
        axs[1, 0].legend()
        plt.show()


if __name__ == '__main__':
    inits = 50
    generations = 50
    path = "ref_comparison/"
    with open('ref_comparison/ref_hamiltonians') as f:
        hdict = eval(f.readline())
    Hc = hdict['Hc']
    Hh2 = hdict['Hh2']

    t1 = time.time()
    ref_comparison(Hc, inits, generations, save_path=path+'Hc_ext', show_figs=False)
    ref_comparison(Hh2, inits, generations, save_path=path+'Hh2_ext', show_figs=False)
    print(time.time() - t1)

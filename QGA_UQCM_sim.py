"""
.. module:: QGA_UQCM.py
    :synopsis: Implements the QGA based on different registers which are pseudo cloned and swapped to implement
    crossover, and semi-sorted to implement selection. This version allows a quantum fitness criteria. Which is
    a fitness criteria defined on a basis which is not known beforehand, although an oracle is needed.
    This GA model is only valid when the fitness can be evaluated by pairs.
    This version implements the cloning subroutine using the Universal Quantum Cloning Machine (UQCM).
.. moduleauthor::  Ruben Ibarrondo (rubenibarrondo@gmail.com)
"""
import os
import numpy as np
import quantum_mats as qm
from copy import deepcopy
from time import time

np.set_printoptions(threshold=np.inf)

# -----------------------------
#
#       QUANTUM GENETIC
#          ALGORITHM
#              -
#        Cloning with
#            UQCM
#
# -----------------------------


def quantum_genetic_algorithm(fitness_criteria, fitness_basis=None,
                              init_population=None, n=None, cl=None,
                              generation_number=100, pm=0.01, mutation_pattern=None, mutation_unitary="x",
                              projection_method="r", pre_projection_unitary="I",
                              store_path=None, track_fidelity=None, track_only_reg_states=True):
    """
    Quantum Genetic Algorithm.
    WARNING! In this version dense matrices are used to represent the density matrix, quantum_mats.rho are not supported
    yet.

    Parameters
    ----------
    fitness_criteria: function
        The criteria evaluated in the auxiliar canonical basis.
            criteria(x,y) = { True, if f(x)<f(y); False, if f(x)>=f(y)}
    fitness_basis: np.array
        The basis in which the fitness_criteria should be evaluated to apply the fitness oracle, each problem basis
        state's coefficients are writen in each column.
        It describes a basis change for the state of a register, from the canonical basis to the problem basis.
        Defaults to None, not basis rotation
    init_population: list, rho, np.array, None
        If list, it is a list of lists describing the classical population,
        must fulfil len(init_population) * len(init_population[0]) = N*cl
        If rho, describes the initial state,
        must fulfil init_population.nq = N*cl.
        If np.array, describes the initial state as a density matrix,
        must fulfil init_population.shape[0] = 2**(N*cl).
        If None, a random state is generated using N and cl.
    n: int
        Number of registers (required). Preferred multiple of four.
    cl: int
        Number of qubits in each register (required). Preferred even.
    generation_number: int
        Number of generation for the genetic algorithm. Defaults to 100.
    pm: float, list
        Probability of bitwise mutation. Defaults to 0.01. Can be overwritten by mutation_pattern.
        If list, it contains the probability of applying each mutation unitary in the mutation_unitary list.
        Probabilities are exclusive, so they must add up to less than one and at most one mutation unitary will be
        applied in each qubit in each generation.
        The length of this array and the mutation_unitary array must match.
    mutation_pattern: list of list of int
        For each generation indicates which qubits to mutate. Defaults to None, using pm=0.01. Overwrites pm.
        Mutation pattern does not support using a set of mutation unitaries for mutation.
    mutation_unitary: np.array, string, list
        If 2x2 np.array matrix describes the mutation unitary.
        If string, one of the following options can be used:
            'not' or 'x' : np.array([[0,1],[1,0]])
            'r': random unitary is generated for each mutation
            'I': identity
            'H': Hadamard gate
        If list, contains a set of mutation unitaries. They will be applied exclusively according to their probability
        determined in the pm array. Both array's length must match.
        Defaults to 'x'.
    projection_method: string
        One of the following options can be used:
            'r': the qubits is randomly projected to 0 or 1, according to the amplitude of each state, and only the
                state according with that value is kept. A pure state would allways yield another pure state after
                projection.
            'ptrace': the partial trace for that qubits is computed and the density matrix is extended to recover the
                original dimensions. This may yield a density matrix not in a pure state. It preserves all the
                statistics related to measurements but it may have a high computational cost.
        Defaults to 'r'.
    pre_projection_unitary:  np.array, string
        If np.array matrix describes the unitary to apply before projection to a register.
        This does not enable changing the acilla measuring basis.
        It can be used to measure in different bases.
        If string, one of the following options can be used:
            'not' or 'x' : np.array([[0,1],[1,0]]) for each qubit in the register
            'r': random unitary is generated for each mutation (not supported yet)
            'I': identity
            'H': Hadamard gate for each qubit in the register
        Defaults to 'I'.
    store_path: string, None
        String describing the path to save the progress.
        If None the progress is not saved.
        Defaults to None.
    track_fidelity: list of states
        List of states for which the fidelity of each registers is tracked.
        Only pure states described with a np.array vector are supported.
        If None the fidelity is not tracked.
        Defaults to None.
    track_only_reg_states: boolean
        True if only the partial density matrix of each register separately should be saved.
        False if the whole density matrix should be saved, this could be too expensive.
        Defaults to True.
    Returns
    -------
    dict:
        Contains the final state, fidelity track...
    """

    # 1 - Get the initial population
    if init_population is None:
        # generate a random population
        raise ValueError('init_population is required in this version. Random population is not supported.')
    elif type(init_population) == qm.rho:
        if not n or not cl:
            raise ValueError('If population is a quantum_matrix.rho object n and cl are required.')
        a = n//2
        ancillas = [ia for ia in range(n*cl, n*cl+a)]
        ancillas_initial_state = np.zeros((2 ** a, 2 ** a))
        ancillas_initial_state[0, 0] = 1
        rho_pop = qm.rho(np.kron(init_population.get_matrix(), ancillas_initial_state), dense=True)
    else:
        population = deepcopy(init_population)
        if n != len(population) or cl != len(population[0]):
            raise ValueError('n and cl mismatch. {}!={} or {}!={}.'.format(n, len(population), cl, len(population[0])))
        i = 0
        index = 0
        for p in population:
            for c in p:
                i += c * 2**(n*cl - index - 1)
                index += 1
        a = n//2
        ancillas = [ia for ia in range(n*cl, n*cl+a)]
        rho_pop = qm.rho([1], [i * 2**a], [i * 2**a], (2**(n*cl + a), 2**(n*cl + a)))

    if track_fidelity:
        fidelity_array = np.zeros(shape=(generation_number, 5, n, len(track_fidelity)))

    if store_path:
        store = open(store_path, 'w')
        print('Genetic Algorithm parameters:', file=store)
        print('generation_numbar = {:d}, pm = {:f}, population number = {:d}, chromosome length = {:d}'.format(
            generation_number, pm, n, cl), file=store)
        print('Tracked fidelity for states:', file=store)
        if track_fidelity:
            for state in track_fidelity:
                if type(state) == qm.rho:
                    print(state.get_matrix(), file=store)
                else:
                    print(repr(state), file=store)
        else:
            print("None", file=store)
        print(file=store)

    # Build some useful matrices for UQCM
    rearrange_mat = np.identity(2 ** (n * cl))
    for k in range(0, n//4):
        s = qm.Swap_reg(range((2 * k + 1) * cl, (2 * k + 2) * cl),
                        range((2 * k + n//2) * cl, (2 * k + n//2 + 1) * cl),
                        n * cl).get_matrix()
        rearrange_mat = np.dot(rearrange_mat, s)

    splus = np.sqrt(2 / (2 ** cl + 1)) * (np.identity(2 ** (2 * cl)) +
                                          qm.Swap_reg(range(0, cl), range(cl, 2 * cl), 2 * cl).get_matrix()) / 2
    splus_mat = splus
    for i in range(n//2-1):
        splus_mat = np.kron(splus_mat, splus)

    # Build the crossover routine unitary
    cross_index = cl // 2
    qq1 = [q1 for nu in range(0, n//2, 2) for q1 in range((n // 2 + nu) * cl + cross_index, (n // 2 + nu + 1) * cl)]
    qq2 = [q2 for nu in range(0, n//2, 2) for q2 in range((n // 2 + nu + 1) * cl + cross_index, (n // 2 + nu + 2) * cl)]
    cross = qm.Swap_reg(qq1, qq2, n*cl)
    cross = qm.KronExpand(1, cross, 2 ** a)
    mat_cross = cross.get_matrix()

    # Build the sort/semi sort step unitary
    # You cant build it completely because you need to set the ancillas to
    # zero in each stage, but this makes it easier later.
    sort_arr = [np.identity(2 ** (n * cl + a)), np.identity(2 ** (n * cl + a))]
    for stage in [0, 1]:
        for reg in range(0, n - 1, 2):
            nu = reg + stage % 2
            if nu + 2 > n:
                continue
            reg1 = list(range(nu * cl, (nu + 1) * cl))
            reg2 = list(range((nu + 1) * cl, (nu + 2) * cl))
            ancilla = n * cl + reg // 2
            orac = qm.Oracle.get_qf_sort_oracle(nq=n * cl + a,
                                                reg1=reg1,
                                                reg2=reg2,
                                                ancilla=ancilla,
                                                uu=fitness_basis,
                                                criteria=fitness_criteria)

            cs = qm.CSwap_reg(ancilla,
                              reg1,
                              reg2,
                              n * cl + a).get_matrix()
            sort_arr[stage] = sort_arr[stage].dot(cs).dot(orac)
    sort0, sort1 = sort_arr
    mat_sort0 = sort0
    mat_sort1 = sort1

    # Build mutation arrays
    if type(mutation_unitary) == list or type(pm) == list:
        if type(mutation_unitary) != list or type(pm) != list:
            raise ValueError("mutation_unitary and pm are not consistent.")
        if len(mutation_unitary) != len(pm):
            raise ValueError("The length of mutation_unitary and pm is not consistent.")
        if sum(pm) >= 1:
            raise ValueError("Mutation probabilities must add up to less than one.")
        if not (mutation_pattern is None):
            raise ValueError("Using mutation_pattern and a set of mutation unitaries is not compatible.")
        use_mutation_unitary_set = True
        pm_sum = sum(pm)
        pm_norm = [pmi / pm_sum for pmi in pm]
    else:
        pm_sum = pm
        use_mutation_unitary_set = False

    if not use_mutation_unitary_set:
        if mutation_unitary not in ["r", "R"]:
            if type(mutation_unitary) == str:
                if mutation_unitary in ["x", "X", "not", "NOT"]:
                    mutation_unitary = qm.rho([1, 1], [0, 1], [1, 0], (2, 2))
                elif mutation_unitary in ["h", "H"]:
                    mutation_unitary = qm.rho(np.array([1, 1, 1, -1])/np.sqrt(2), [0, 0, 1, 1], [0,  1, 0, 1], (2, 2))
                elif mutation_unitary in ["i", "I"]:
                    mutation_unitary = qm.Identity(2)
                else:
                    raise ValueError("mutation_unitary = \"%s\" not recognised." % mutation_unitary)
            else:
                mutation_unitary = qm.rho(mutation_unitary, dense=True)

            mut_arr = []
            mat_mut_arr = []
            for i in range(n * cl):
                mut_arr.append(qm.KronExpand(2 ** i, mutation_unitary, 2 ** (n * cl - i - 1 + a)))
                mat_mut_arr.append(np.kron(np.kron(np.identity(2 ** i),
                                                   mutation_unitary.get_matrix()),
                                           np.identity(2 ** (n * cl - i - 1 + a))))

    # Build the pre-projection rotation
    if type(pre_projection_unitary) == str:
        if pre_projection_unitary in ["i", "I"]:
            pre_projection_unitary = "I"
        else:
            raise ValueError("pre_projection_unitary = \"%s\" not supported." % pre_projection_unitary)
    else:
        lower_rot_mat = np.identity(2 ** (n//2 * cl))
        for lreg in range(n-n//2):
            lower_rot_mat = np.kron(lower_rot_mat, pre_projection_unitary)
        lower_rot_mat = np.kron(lower_rot_mat, np.identity(2 ** a))

        lower_rot = None  # Not implemented

    for generation in range(generation_number):
        if track_fidelity:
            for reg in range(n):
                reg_state = rho_pop.partial_trace(list(range(reg * cl, (reg + 1) * cl)))
                for i, state in enumerate(track_fidelity):
                    fidelity_array[generation, 0, reg, i] = reg_state.fidelity(state)
        if store_path:
            print('-'*32, '\n',
                  ' '*9, 'Generation: {}'.format(generation),
                  '\n', '-' * 32, '\n',
                  file=store, sep='')

            if not track_only_reg_states:
                print('initial state, population + ancillas:\n', repr(rho_pop.store), file=store)
                print(file=store)
            else:
                print('initial state:', file=store)
                for reg in range(n):
                    print('register {:2d}'.format(reg), end=' '*4, file=store)
                    print(repr(rho_pop.partial_trace(list(range(reg * cl, (reg + 1) * cl))).get_matrix()), file=store)
                    print('\n', file=store)

        # 2 - sort or semi sort:
        for stage in range(0, n):
            sort = [sort0, sort1][stage % 2]
            mat_sort = [mat_sort0, mat_sort1][stage % 2]
            rho_pop = qm.rho(mat_sort.dot(rho_pop.get_matrix()).dot(np.transpose(mat_sort).conjugate()), dense=True)

            if projection_method != 'ptrace':
                for ai in ancillas:
                    rho_pop.projection_controlled_rotation(ai, 1, qm.rho([1, 1], [0, 1], [1, 0], (2, 2)),
                                                           pre_projection_unitary, projection_method)
            else:
                rho_pop = rho_pop.partial_trace(list(range(n * cl)))
                rho_pop = qm.rho(np.kron(rho_pop.get_matrix(),
                                         qm.rho([1], [0], [0], (2 ** a, 2 ** a)).get_matrix()), dense=True)

        if track_fidelity:
            for reg in range(n):
                reg_state = rho_pop.partial_trace(list(range(reg * cl, (reg + 1) * cl)))
                for i, state in enumerate(track_fidelity):
                    fidelity_array[generation, 1, reg, i] = reg_state.fidelity(state)
        if store_path:
            if not track_only_reg_states:
                print('sorted state, population + ancillas:\n', repr(rho_pop.store), file=store)
            else:
                print('sorted state:', file=store)
                for reg in range(n):
                    print('register {:2d}'.format(reg), end=' ' * 4, file=store)
                    print(repr(rho_pop.partial_trace(list(range(reg * cl, (reg + 1) * cl))).get_matrix()), file=store)
                    print('\n', file=store)
            print(file=store)

        # 3 - Clear the lowest half
        if projection_method != 'ptrace':
            for q in range(n//2 * cl, n * cl):
                rho_pop.projection_controlled_rotation(q, 1, qm.rho([1, 1], [0, 1], [1, 0], (2, 2)),
                                                       pre_projection_unitary, projection_method)
        else:
            if type(pre_projection_unitary) != str:
                rho_pop = qm.rho(lower_rot_mat.dot(rho_pop.get_matrix()).dot(np.transpose(np.conjugate(lower_rot_mat))), dense=True)
            elif pre_projection_unitary != "I":
                raise Exception("Only \"I\" pre_projection_unitary is supported with type str")

            rho_pop = rho_pop.partial_trace(list(range(n//2 * cl)))
            rho_pop = qm.rho(np.kron(rho_pop.get_matrix(),
                                     qm.rho([1], [0], [0],
                                            (2 ** (n*cl - n//2 * cl + a),
                                             2 ** (n*cl - n//2 * cl + a))).get_matrix()), dense=True)
        if track_fidelity:
            for reg in range(n):
                reg_state = rho_pop.partial_trace(list(range(reg * cl, (reg + 1) * cl)))
                for i, state in enumerate(track_fidelity):
                    fidelity_array[generation, 2, reg, i] = reg_state.fidelity(state)
        if store_path:
            if not track_only_reg_states:
                print('state after sort and clear, population + ancillas:\n', repr(rho_pop.store), file=store)
            else:
                print('state after sort and clear:', file=store)
                for reg in range(n):
                    print('register {:2d}'.format(reg), end=' ' * 4, file=store)
                    print(repr(rho_pop.partial_trace(list(range(reg * cl, (reg + 1) * cl))).get_matrix()), file=store)
                    print('\n', file=store)
            print(file=store)

        # 4 - Crossover to fill
        # 4.a Put the lower registers in a NOT-normalized maximally mixed state
        rho_pop_mat = np.kron(rho_pop.partial_trace(range(n//2 * cl)).get_matrix(),
                              np.identity(2**(n//2 * cl)))
        # 4.b Rearrange their positions so that they are in the appropriate order for clone
        rho_pop_mat = np.dot(np.dot(rearrange_mat, rho_pop_mat), rearrange_mat.transpose().conjugate())
        # 4.c Apply Splus matrix (projection+normalization into the symmetric subspace)
        rho_pop_mat = np.dot(np.dot(splus_mat, rho_pop_mat), splus_mat.transpose().conjugate())
        # 4.d Rearrage their positions so that they are in their original place
        rho_pop_mat = np.dot(np.dot(rearrange_mat, rho_pop_mat), rearrange_mat.transpose().conjugate())
        rho_pop_mat /= np.trace(rho_pop_mat)  # although small, there is a numerical error
        rho_pop = qm.rho(np.kron(rho_pop_mat,
                                 qm.rho([1], [0], [0], (2 ** a, 2 ** a)).get_matrix()), dense=True)
        rho_pop = qm.rho(mat_cross.dot(rho_pop.get_matrix()).dot(np.transpose(mat_cross)), dense=True)

        if track_fidelity:
            for reg in range(n):
                reg_state = rho_pop.partial_trace(list(range(reg * cl, (reg + 1) * cl)))
                for i, state in enumerate(track_fidelity):
                    fidelity_array[generation, 3, reg, i] = reg_state.fidelity(state)
        if store_path:
            if not track_only_reg_states:
                print('state after crossover, population + ancillas:\n', repr(rho_pop.store), file=store)
            else:
                print('state after crossover:', file=store)
                for reg in range(n):
                    print('register {:2d}'.format(reg), end=' ' * 4, file=store)
                    print(repr(rho_pop.partial_trace(list(range(reg * cl, (reg + 1) * cl))).get_matrix()), file=store)
                    print('\n', file=store)
            print(file=store)

        # 5 - mutate with probability pm
        for c in range(n * cl):
            r = np.random.random()
            if r < pm_sum:
                mu = mutation_unitary[np.random.choice(range(len(pm)), p=pm_norm)]
                mut_mat = np.kron(np.kron(np.identity(2 ** c), mu),
                                  np.identity(2 ** (n * cl - c - 1 + a)))
                rho_pop = qm.rho(mut_mat.dot(rho_pop.get_matrix().dot(np.transpose(mut_mat).conjugate())),
                                 dense=True)

        if track_fidelity:
            for reg in range(n):
                reg_state = rho_pop.partial_trace(list(range(reg * cl, (reg + 1) * cl)))
                for i, state in enumerate(track_fidelity):
                    fidelity_array[generation, 4, reg, i] = reg_state.fidelity(state)
        if store_path:
            if not track_only_reg_states:
                print('state after mutation, population + ancillas:\n', repr(rho_pop.store), file=store)

            else:
                print('state after mutation:', file=store)
                for reg in range(n):
                    print('register {:2d}'.format(reg), end=' ' * 4, file=store)
                    print(repr(rho_pop.partial_trace(list(range(reg * cl, (reg + 1) * cl))).get_matrix()), file=store)
                    print('\n', file=store)
            print(file=store)

    if store_path:
        store.close()
    if track_fidelity:
        return rho_pop, fidelity_array
    else:
        return rho_pop


# MAIN EXAMPLE


def qga_uqcm_test(fitness_states, samples, dirpath):
    """
    Not updated to QGA_UQCM.
    :return:
    """
    g = 10
    n = 4
    cl = 2
    pm = [1 / n / cl / 3] * 3
    ppu = "I"
    mu = [np.array([[0, 1], [1, 0]]),
          np.array([[0, -1j], [1j, 0]]),
          np.array([[1, 0], [0, -1]])]
    new_dirpath = dirpath

    i = 1
    while os.path.exists(new_dirpath) and i < 1000:
        if dirpath.split("_")[-1].isnumeric():
            if i == 1:
                i = int(dirpath.split("_")[-1]) + 1
                new_dirpath = "_".join(dirpath.split("_")[:-1]) + ("_%03d" % i)
            else:
                new_dirpath = "_".join(dirpath.split("_")[:-1]) + ("_%03d" % i)
        else:
            new_dirpath = dirpath + ("_%03d" % i)
        i += 1
    if i > 1:
        print("dirpath collision, changed to:\n%s" % new_dirpath)
    dirpath = new_dirpath
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)
    else:
        raise Exception("dirpath should be new, dirpath collision not corrected.")

    uu = np.zeros((4, 4), dtype=np.complex64)
    for i, state in enumerate(fitness_states):
        uu[:, i] = state
    criteria = lambda x, y: sum(int(xi) * 2 ** (len(x) - i - 1) for i, xi in enumerate(x)) > sum(
        int(yi) * 2 ** (len(y) - i - 1) for i, yi in enumerate(y))
    tf = fitness_states

    with open(dirpath+'/0-Notes', 'w') as notes:
        notes.write(dirpath + "\n-" * 32 + "\n" +
                    "Simulates QGA with UQCM and with fitness criteria based in this sorting order:\n")
        for i, state in enumerate(fitness_states):
            notes.write("%d.\n" % i)
            notes.write(repr(state))
            notes.write("\n")
        notes.write('Genetic Algorithm parameters:\n')
        notes.write('generation_numbar = {:d}\n population number = {:d}\n chromosome length = {:d}\n'.format(g, n, cl))
        notes.write('pm:\n')
        notes.write(repr(pm))
        notes.write("\n")
        notes.write('pre_projection_unitary:\n')
        notes.write(repr(ppu))
        notes.write("\n")
        notes.write('mutation_unitary:\n')
        notes.write(repr(mu))
        notes.write("\n")

    for trial in range(samples):
        print("trial ", trial, end=' ')
        t1 = time()

        rho_population = qm.rho.gen_random_rho(n * cl)

        rho_final, ft = quantum_genetic_algorithm(criteria, fitness_basis=uu,
                                                  init_population=rho_population, n=n, cl=cl,
                                                  generation_number=g, pm=pm, mutation_unitary=mu,
                                                  projection_method="ptrace", pre_projection_unitary=ppu,
                                                  store_path=None,
                                                  track_fidelity=tf)
        print(time() - t1)

        with open(dirpath+'/fidelity_tracks_{:03d}'.format(trial), 'w') as file:
            file.write("Tracking fidelities for:\n")
            for i, state in enumerate(fitness_states):
                file.write("%d.\n" % i)
                file.write(repr(state))
                file.write("\n")
            file.write("\n")
            file.write(repr(ft))


if __name__ == '__main__':
    from scipy.stats import special_ortho_group
    state_case_number = 100
    samples = 2
    run_num = 4
    dirpath = 'QGA_UQCM_run_{:02d}/QGA_UQCM_test_'.format(run_num)

    for state_case in range(state_case_number):
        print("State case: %d" % state_case)
        if state_case == 0:
            tf = [np.array([1, 0, 0, 0]),
                  np.array([0, 1, 0, 0]),
                  np.array([0, 0, 1, 0]),
                  np.array([0, 0, 0, 1])]
        elif state_case == 1:
            tf = [np.full(4, 1 / 2),
                  np.full(4, 1 / 2) * np.array([1, -1, 1, -1]),
                  np.full(4, 1 / 2) * np.array([1, 1, -1, -1]),
                  np.full(4, 1 / 2) * np.array([1, -1, -1, 1])]
        else:
            # DOES NOT GENERATE COMPLEX NUMBERS!!!
            ortho_group = special_ortho_group.rvs(4)
            tf = [ortho_group[:, i] for i in range(4)]

        qga_uqcm_test(fitness_states=tf, samples=samples, dirpath=dirpath+("%03d" % (state_case + 1)))

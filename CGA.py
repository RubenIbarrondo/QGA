"""
.. module:: CGA.py
    :synopsis: Implements the CGA_1, a classical GA that encodes the individuals as a 2^c-element vectors. Crossover and
    mutations vary.
.. moduleauthor::  Ruben Ibarrondo (rubenibarrondo@gmail.com)
"""
import numpy as np
import QGA_UQCM_sim as uqcm
import QGA_BCQO_sim as bcqo

# Define some useful matrices
X = np.array([[0,1], [1,0]])
Y = np.array([[0,-1j], [1j,0]])
Z = np.array([[1,0],[0,-1]])
I = np.identity(2)


# Define the fitness evaluating function
def energy(individual, Hamiltonian):
    return np.dot(individual.conjugate().transpose(), np.dot(Hamiltonian, individual))


def get_eigenbasis(Hamiltonian):
    energies, eigenstates = np.linalg.eigh(Hamiltonian)
    energies_states = sorted([(e, sind) for sind, e in enumerate(energies)])
    eigenstates = [eigenstates[:, sind] for e, sind in energies_states]
    return energies, eigenstates


# Define different crossover methods
def crossover_a(individual1, individual2):
    child1 = 2 * individual1 + individual2
    child1 = child1 / np.linalg.norm(child1)
    child2 = individual1 + 2 * individual2
    child2 = child2 / np.linalg.norm(child2)
    return child1, child2


def crossover_b(individual1, individual2):
    d = len(individual1)
    child1 = np.copy(individual1)
    child1[d//2:] = individual2[d//2:]
    child2 = np.copy(individual2)
    child2[d // 2:] = individual1[d // 2:]
    return child1, child2


# Define different mutation methods
def mutation_i(individual, pm):
    child = np.copy(individual)
    r = np.random.random(child.shape)
    sigma = 0.228 * (r <= pm)
    dr = np.random.normal(0, sigma, child.shape)
    child = child + dr
    child = child / np.linalg.norm(child)
    return child


def mutation_ii(individual, pm):
    child = np.copy(individual)
    c = int(np.round(np.log2(len(child))))
    r = np.random.random(c)

    mutmat = 1
    for rq in r:
        if rq <= pm / 3:
            mutmat = np.kron(mutmat, X)
        elif rq <= 2 * pm / 3:
            mutmat = np.kron(mutmat, Y)
        elif rq <= pm:
            mutmat = np.kron(mutmat, Z)
        else:
            mutmat = np.kron(mutmat, I)


    child = np.dot(mutmat, child)
    child = child / np.linalg.norm(child)
    return child


def fidelity(eigenstate, individual):
    return abs(np.dot(eigenstate.conjugate().transpose(), individual))


def CGA_1(Hamiltonian, int_population, n, cl, generation_number, pm,
          crossover_method, mutation_method):
    """Real valued cGA."""
    assert (n % 4 == 0) and (cl % 2 == 0), "n must be divisible by 4, cl must be even."
    assert (int(np.round(np.log2(len(Hamiltonian)))) == cl), "concordance in the number of qubits"
    population = int_population

    energies, eigenstates = get_eigenbasis(Hamiltonian)

    fidelity_track = np.empty((generation_number+1, n, 2**cl))
    fidelity_track[0, :, :] = np.array([[fidelity(eigenstate, individual)
                                        for eigenstate in eigenstates]
                                        for individual in population])

    for generation in range(generation_number):

        # Selection
        energy_individual = [(energy(individual, Hamiltonian), i) for i, individual in enumerate(population)]
        energy_individual = sorted(energy_individual)
        population = [population[i] for e, i in energy_individual]
        fidelity_track[generation+1, :, :] = np.array([[fidelity(eigenstate, individual)
                                                       for eigenstate in eigenstates]
                                                       for individual in population])

        population[n//2:] = [None] * (n//2)

        # Crossover
        for i in range(n//4):
            population[n//2+2*i], population[n//2+2*i+1] = crossover_method(population[2*i], population[2*i+1])

        # Mutation
        for i, individual in enumerate(population):
            population[i] = mutation_method(individual, pm)

    return population, fidelity_track


def pop2bin(population):
    c = int(np.round(np.log2(len(population[0]))))
    pop_num = [int(np.where(state >= 1-1e-6)[0]) if state is not None else None for state in population]
    pop_bin = []
    for num in pop_num:
        if num is not None:
            b = bin(num)[2:]
            if len(b) < c:
                b = '0'*(c-len(b)) + b
            pop_bin.append([bi for bi in b])
        else:
            pop_bin.append(None)
    return pop_bin


def bin2pop(population_bin):
    c = len(population_bin[0])
    pop = [np.zeros(2**c) for ind in population_bin]
    for ind, binind in zip(pop, population_bin):
        num = int(np.sum([int(b)*2**(c-bi-1) for bi, b in enumerate(binind)]))
        ind[num] = 1
    return pop


def CGA_2(Hamiltonian, int_population, n, cl, generation_number, pm,
          crossover_method, mutation_method):
    """Binary cGA."""
    assert (n % 4 == 0) and (cl % 2 == 0), "n must be divisible by 4, cl must be even."
    assert (int(np.round(np.log2(len(Hamiltonian)))) == cl), "concordance in the number of qubits"
    assert np.sum(np.abs(Hamiltonian - np.diag(np.diag(Hamiltonian)))) < 1e-6, "The Hamiltonian must be diagonal for CGA_2"

    if type(int_population[0]) == np.ndarray:
        if np.any([np.sum(state) < 1-1e-6 for state in int_population]):
            raise Warning("States casted to binary representation.")
            population = [1 * (state == np.max(state)) for state in int_population]
        else:
            population = int_population
    else:
        population = bin2pop(int_population)

    energies, eigenstates = get_eigenbasis(Hamiltonian)

    fidelity_track = np.empty((generation_number+1, n, 2**cl))
    fidelity_track[0, :, :] = np.array([[fidelity(eigenstate, individual)
                                        for eigenstate in eigenstates]
                                        for individual in population])

    for generation in range(generation_number):

        # Selection
        energy_individual = [(energy(individual, Hamiltonian), i) for i, individual in enumerate(population)]
        energy_individual = sorted(energy_individual)
        population = [population[i] for e, i in energy_individual]
        fidelity_track[generation+1, :, :] = np.array([[fidelity(eigenstate, individual)
                                                       for eigenstate in eigenstates]
                                                       for individual in population])

        population[n//2:] = [None] * (n//2)

        # Cast population to binary
        population_bin = pop2bin(population)

        # Crossover
        for i in range(n//4):
            population_bin[n//2+2*i] = population_bin[2*i][:cl//2] + population_bin[2*i+1][cl//2:]
            population_bin[n//2+2*i+1] = population_bin[2*i+1][:cl//2] + population_bin[2*i][cl//2:]

        # Mutation
        for i, individual in enumerate(population):
            for bit in range(cl):
                r = np.random.random()
                if r <= pm:
                    population_bin[i][bit] = str(int(not bool(population_bin[i][bit])))

        # Cast population to states
        population = bin2pop(population_bin)

    return population, fidelity_track


def CGA_3(Hamiltonian, int_population, n, cl, generation_number, pm,
          crossover_method, mutation_method):
    """Traditional QIGA.
    References:
        Han, K.-H.; Kim, J.-H. (2002). Quantum-Inspired Evolutionary Algorithm for a Class of Combinatorial Optimization. IEEE Transactions on Evolutionary Computation, 6 (6), 580–593. https://doi.org/10.1080/01496395.2017.1297456
        Narayanan, A.; Moore, M. (1996). Quantum-inspired genetic algorithms. Proceedings of IEEE International Conference on Evolutionary Computation, 61–66. https://doi.org/10.1109/ICEC.1996.542334
    ¿?¿?¿?
    """

    raise NotImplementedError("QIGA is not yet supported.")

    assert (n % 4 == 0) and (cl % 2 == 0), "n must be divisible by 4, cl must be even."
    assert (int(np.round(np.log2(len(Hamiltonian)))) == cl), "concordance in the number of qubits"
    assert np.sum(np.abs(Hamiltonian - np.diag(np.diag(Hamiltonian)))) < 1e-6, "The Hamiltonian must be diagonal for CGA_2"

    # Check if population is received in state form or in binary form
    population = int_population

    energies, eigenstates = get_eigenbasis(Hamiltonian)

    fidelity_track = np.empty((generation_number+1, n, 2**cl))
    fidelity_track[0, :, :] = np.array([[fidelity(eigenstate, individual)
                                        for eigenstate in eigenstates]
                                        for individual in population])

    for generation in range(generation_number):

        # Selection
        energy_individual = [(energy(individual, Hamiltonian), i) for i, individual in enumerate(population)]
        energy_individual = sorted(energy_individual)
        population = [population[i] for e, i in energy_individual]
        fidelity_track[generation+1, :, :] = np.array([[fidelity(eigenstate, individual)
                                                       for eigenstate in eigenstates]
                                                       for individual in population])

        population[n//2:] = [None] * (n//2)

        # Cast population to binary
        population_bin = pop2bin(population)

        # Crossover
        for i in range(n//4):
            population_bin[n//2+2*i] = population_bin[2*i][:cl//2] + population_bin[2*i+1][cl//2:]
            population_bin[n//2+2*i+1] = population_bin[2*i+1][:cl//2] + population_bin[2*i][cl//2:]

        # Mutation
        for i, individual in enumerate(population):
            for bit in range(cl):
                r = np.random.random()
                if r <= pm:
                    population_bin[i][bit] = str(int(not bool(population_bin[i][bit])))

        # Cast population to states
        population = bin2pop(population_bin)

    return population, fidelity_track


if __name__ == '__main__':
    import quantum_mats as qm
    import matplotlib.pyplot as plt

    n = 4
    cl = 2

    init_pop = [qm.rho.gen_random_rho(2, asvector=True) for i in range(4)]
    bin_init_pop = [1 * (state == np.max(state)) for state in init_pop]

    generations = 10
    pm = 0.125
    f = open('ref_comparison/ref_hamiltonians')
    hdict = eval(f.readline())
    f.close()

    fig, axs = plt.subplots(2, 2, figsize=[15, 6])

    # Classical Hc
    H = hdict['Hc']
    pop, ftai = CGA_1(H, init_pop, 4, 2, generations, pm, crossover_a, mutation_i)
    pop, ftbi = CGA_1(H, init_pop, 4, 2, generations, pm, crossover_b, mutation_i)
    pop, ftaii = CGA_1(H, init_pop, 4, 2, generations, pm, crossover_a, mutation_ii)
    pop, ftbii = CGA_1(H, init_pop, 4, 2, generations, pm, crossover_b, mutation_ii)
    pop, ft2 = CGA_2(H, bin_init_pop, 4, 2, generations, pm, None, None)

    axs[0, 0].set_title("Hc")
    axs[0, 0].plot(ftai[:, 0, 0], label='1.a-i')
    axs[0, 0].plot(ftbi[:, 0, 0], label='1.b-i')
    axs[0, 0].plot(ftaii[:, 0, 0], label='1.a-ii')
    axs[0, 0].plot(ftbii[:, 0, 0], label='1.b-ii')
    axs[0, 0].plot(ft2[:, 0, 0], label='2.')
    axs[0, 0].legend()

    # Classical Hh2
    axs[0, 1].set_title("Hh2")
    H = hdict['Hh2']
    pop, ftai = CGA_1(H, init_pop, 4, 2, generations, pm, crossover_a, mutation_i)
    pop, ftbi = CGA_1(H, init_pop, 4, 2, generations, pm, crossover_b, mutation_i)
    pop, ftaii = CGA_1(H, init_pop, 4, 2, generations, pm, crossover_a, mutation_ii)
    pop, ftbii = CGA_1(H, init_pop, 4, 2, generations, pm, crossover_b, mutation_ii)

    axs[0, 1].plot(ftai[:, 0, 0], label='1.a-i')
    axs[0, 1].plot(ftbi[:, 0, 0], label='1.b-i')
    axs[0, 1].plot(ftaii[:, 0, 0], label='1.a-ii')
    axs[0, 1].plot(ftbii[:, 0, 0], label='1.b-ii')
    axs[0, 1].legend()

    # Quantum Stuff
    criteria = lambda x, y: sum(int(xi) * 2 ** (len(x) - i - 1) for i, xi in enumerate(x)) > sum(
        int(yi) * 2 ** (len(y) - i - 1) for i, yi in enumerate(y))
    ppu = "I"
    mu = [np.array([[0, 1], [1, 0]]),
          np.array([[0, -1j], [1j, 0]]),
          np.array([[1, 0], [0, -1]])]
    pm_arr = [1 / n / cl / 3] * 3

    # Difference wrt our other tests, the initial states are separable!
    #rho_population = qm.rho.gen_random_rho(n * cl)
    #raise Warning("Update this so that it uses the same initial state!")
    rho_vector = init_pop[0]
    for state in init_pop[1:]:
        rho_vector = np.kron(rho_vector, state)
    rho_vector = rho_vector / np.linalg.norm(rho_vector)
    mat_form = np.kron(rho_vector.reshape((rho_vector.shape[0], 1)), rho_vector.conjugate())
    rho_population = qm.rho.gen_rho_from_matrix(mat_form)

    # Quantum Hc
    H = hdict['Hc']
    energies, eigenstates = get_eigenbasis(H)
    uu = np.zeros((4, 4), dtype=complex)
    for i, state in enumerate(eigenstates):
        uu[:, i] = state
    tf = eigenstates

    rho_final, ftunm = uqcm.quantum_genetic_algorithm(criteria, fitness_basis=uu,
                                                      init_population=rho_population, n=n, cl=cl,
                                                      generation_number=generations, pm=0, mutation_unitary='I',
                                                      projection_method="ptrace", pre_projection_unitary=ppu,
                                                      store_path=None,
                                                      track_fidelity=tf)

    rho_final, ftuwm = uqcm.quantum_genetic_algorithm(criteria, fitness_basis=uu,
                                                      init_population=rho_population, n=n, cl=cl,
                                                      generation_number=generations, pm=pm_arr, mutation_unitary=mu,
                                                      projection_method="ptrace", pre_projection_unitary=ppu,
                                                      store_path=None,
                                                      track_fidelity=tf)

    rho_final, ftbnm = bcqo.quantum_genetic_algorithm(criteria, fitness_basis=uu,
                                                      init_population=rho_population, n=n, cl=cl,
                                                      generation_number=generations, pm=0, mutation_unitary='I',
                                                      projection_method="ptrace", pre_projection_unitary=ppu,
                                                      store_path=None,
                                                      track_fidelity=tf)

    rho_final, ftbwm = bcqo.quantum_genetic_algorithm(criteria, fitness_basis=uu,
                                                      init_population=rho_population, n=n, cl=cl,
                                                      generation_number=generations, pm=pm_arr, mutation_unitary=mu,
                                                      projection_method="ptrace", pre_projection_unitary=ppu,
                                                      store_path=None,
                                                      track_fidelity=tf)
    axs[1, 0].set_title("Hc")
    axs[1, 0].plot(ftunm[:, 1, 0, 0], label='uqcm-nm')
    axs[1, 0].plot(ftbnm[:, 1, 0, 0], label='bcqo-nm')
    axs[1, 0].plot(ftuwm[:, 1, 0, 0], label='uqcm-wm')
    axs[1, 0].plot(ftbwm[:, 1, 0, 0], label='bcqo-wm')
    axs[1, 0].legend()

    # Quantum Hh2
    H = hdict['Hh2']
    energies, eigenstates = get_eigenbasis(H)
    uu = np.zeros((4, 4), dtype=complex)
    for i, state in enumerate(eigenstates):
        uu[:, i] = state
    tf = eigenstates

    rho_final, ftunm = uqcm.quantum_genetic_algorithm(criteria, fitness_basis=uu,
                                                      init_population=rho_population, n=n, cl=cl,
                                                      generation_number=generations, pm=0, mutation_unitary='I',
                                                      projection_method="ptrace", pre_projection_unitary=ppu,
                                                      store_path=None,
                                                      track_fidelity=tf)

    rho_final, ftuwm = uqcm.quantum_genetic_algorithm(criteria, fitness_basis=uu,
                                                      init_population=rho_population, n=n, cl=cl,
                                                      generation_number=generations, pm=pm_arr, mutation_unitary=mu,
                                                      projection_method="ptrace", pre_projection_unitary=ppu,
                                                      store_path=None,
                                                      track_fidelity=tf)
    rho_final, ftbnm = bcqo.quantum_genetic_algorithm(criteria, fitness_basis=uu,
                                                      init_population=rho_population, n=n, cl=cl,
                                                      generation_number=generations, pm=0, mutation_unitary='I',
                                                      projection_method="ptrace", pre_projection_unitary=ppu,
                                                      store_path=None,
                                                      track_fidelity=tf)

    rho_final, ftbwm = bcqo.quantum_genetic_algorithm(criteria, fitness_basis=uu,
                                                      init_population=rho_population, n=n, cl=cl,
                                                      generation_number=generations, pm=pm_arr, mutation_unitary=mu,
                                                      projection_method="ptrace", pre_projection_unitary=ppu,
                                                      store_path=None,
                                                      track_fidelity=tf)
    axs[1, 1].set_title("Hh2")
    axs[1, 1].plot(ftunm[:, 1, 0, 0], label='uqcm-nm')
    axs[1, 1].plot(ftbnm[:, 1, 0, 0], label='bcqo-nm')
    axs[1, 1].plot(ftuwm[:, 1, 0, 0], label='uqcm-wm')
    axs[1, 1].plot(ftbwm[:, 1, 0, 0], label='bcqo-wm')
    axs[1, 1].legend()

    plt.show()

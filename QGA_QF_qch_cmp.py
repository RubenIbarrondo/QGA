import quantum_mats as qm
import numpy as np
from scipy import linalg
import QGA_QF


if __name__ == '__main__':
    g = 5
    n = 4
    cl = 2
    pm = 0

    u = np.array([[0.29419042, 0.27876697, -0.66839818, 0.6236865],
                  [0.45200546, -0.83398993, -0.28231706, -0.14299979],
                  [0.45433315, 0.46058814, -0.27247138, -0.71217927],
                  [-0.70903064, -0.12086497, -0.63190213, -0.28873328]])
    track_fidelity = [u[:, i] for i in range(u.shape[1])]

    criteria = lambda x, y: sum(int(xi) * 2 ** (len(x) - i - 1) for i, xi in enumerate(x)) > sum(
        int(yi) * 2 ** (len(y) - i - 1) for i, yi in enumerate(y))

    A = np.load("quantum_channel_analysis_run2-5/fixed_point_1.npy")
    A2 = np.load("quantum_channel_analysis2_run2-5/fixed_point_1.npy")
    print("max|A-A2| = ", np.max(abs(A-A2)))
    print("F(A,A2) = ", abs(np.trace(linalg.sqrtm(linalg.sqrtm(A2) @ A @ linalg.sqrtm(A2))))**2)

    A_post = QGA_QF.quantum_genetic_algorithm(fitness_criteria=criteria, fitness_basis=u,
                                              init_population=qm.rho(A, dense=True), n=n, cl=cl,
                                              generation_number=5, pm=0, projection_method='ptrace')
    A_post = A_post.partial_trace(list(range(n * cl))).get_matrix()

    print("max|A-A_post| = ", np.max(abs(A - A_post)))
    print("F(A,A_post) = ", abs(np.trace(linalg.sqrtm(linalg.sqrtm(A_post) @ A @ linalg.sqrtm(A_post))))**2)

    print("\nA:")

    rho_pop = qm.rho(A, dense=True)

    fidelity_array = np.zeros((n, len(track_fidelity)))

    for reg in range(n):
        reg_state = rho_pop.partial_trace(list(range(reg * cl, (reg + 1) * cl)))
        print("reg {:d}    ".format(reg), end="")
        for i, state in enumerate(track_fidelity):
            fidelity_array[reg, i] = reg_state.fidelity(state)
            print("{:5.3f}".format(fidelity_array[reg, i]), end="\t")
        print()

    print("\nA_post:")

    rho_pop = qm.rho(A_post, dense=True)

    fidelity_array = np.zeros((n, len(track_fidelity)))

    for reg in range(n):
        reg_state = rho_pop.partial_trace(list(range(reg * cl, (reg + 1) * cl)))
        print("reg {:d}    ".format(reg), end="")
        for i, state in enumerate(track_fidelity):
            fidelity_array[reg, i] = reg_state.fidelity(state)
            print("{:5.3f}".format(fidelity_array[reg, i]), end="\t")
        print()


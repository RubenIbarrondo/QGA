import quantum_mats as qm
#import matplotlib.pyplot as plt
import itertools
import numpy as np
import scipy.sparse
import scipy.linalg as linalg
from scipy.stats import special_ortho_group
import time


def get_fixed_points_after_crossover(u=None, show_progress=True):
    g = 5
    n = 4
    cl = 2
    pm = 1 / n / cl

    # SORT
    a = int(np.ceil(n / 2) * (n - 1))

    criteria = lambda x, y: sum(int(xi) * 2 ** (len(x) - i - 1) for i, xi in enumerate(x)) > sum(
        int(yi) * 2 ** (len(y) - i - 1) for i, yi in enumerate(y))

    # u = special_ortho_group.rvs(2 ** cl)
    # u = np.array([[0.29419042, 0.27876697, -0.66839818, 0.6236865],
    #              [0.45200546, -0.83398993, -0.28231706, -0.14299979],
    #              [0.45433315, 0.46058814, -0.27247138, -0.71217927],
    #              [-0.70903064, -0.12086497, -0.63190213, -0.28873328]])
    if u is None:
        u = np.identity(4)

    U = np.kron(u, u)
    U = np.kron(U, U)

    start = time.time()
    sort_unitary = qm.Identity(2 ** (n * cl + a))
    for stage in range(n):
        for reg in range(0, n - 1, 2):
            nu = reg + stage % 2
            if nu + 2 > n:
                continue
            reg1 = list(range(nu * cl, (nu + 1) * cl))
            reg2 = list(range((nu + 1) * cl, (nu + 2) * cl))
            ancilla = n * cl + (stage // 2) * (n - 1) + (stage % 2) * (n // 2) + reg // 2
            orac = qm.Oracle(criteria=criteria,
                             nq=n * cl + a,
                             reg1=reg1,
                             reg2=reg2,
                             ancilla=ancilla)

            cs = qm.CSwap_reg(ancilla,
                              reg1,
                              reg2,
                              n * cl + a)
            sort_unitary = cs.dot(orac).dot(sort_unitary)

    Es_arr = []
    for k in range(2 ** a):
        kbin = bin(k)[2:]
        if len(kbin) < a:
            kbin = "0" * (a - len(kbin)) + kbin

        kvalid = True
        s0us1 = [reg // 2 for reg in range(0, n - 1, 2)] + [(n // 2) + reg // 2 for reg in range(0, n - 2, 2)]
        gt = int(any(kbin[j] == "1" for j in s0us1))
        for stage in range(2, n):
            st = [(stage // 2) * (n - 1) + (stage % 2) * (n // 2) + reg // 2 for reg in range(0, n - 1 - stage % 2, 2)]
            ft = int(any(kbin[j] == "1" for j in st))
            if gt < ft:
                kvalid = False
                break

        if not kvalid:
            continue

        Ek = np.zeros((2 ** (n * cl), 2 ** (n * cl)))
        for i in range(2 ** (n * cl)):
            # for j in range(2 ** (n * cl)):
            #    Ek[i, j] = sort_unitary.get(k + i * 2 ** a, j * 2 ** a)
            for j, element in sort_unitary.row(k + i * 2 ** a):
                if j % (2 ** a) == 0:
                    Ek[i, j // (2 ** a)] = element
        if np.max(Ek) < 1e-5:
            continue

        Es_arr.append(U @ Ek @ U.transpose().conjugate())

    # RESET
    Er_arr = []
    for k in range(2 ** (n // 2 * cl)):
        Pk = np.zeros((2 ** (n // 2 * cl), 2 ** (n // 2 * cl)))
        Pk[0, k] = 1

        Ek = np.kron(np.identity(2 ** (n // 2 * cl)), Pk)
        Er_arr.append(Ek)

    # CLONE
    clone = qm.Identity(2 ** (n * cl))
    for nu in range(0, n // 2):
        s = qm.Swap_reg(range((nu + 1) * cl, (nu + 2) * cl),
                        range((nu + n // 2) * cl, (nu + 1 + n // 2) * cl),
                        n * cl)
        uc = qm.Uclone(2 ** cl)
        unu = s.dot(qm.KronExpand(2 ** (nu * cl), uc, 2 ** ((n - nu - 2) * cl))).dot(s)
        clone = clone.dot(unu)
    mat_clone = clone.get_matrix()

    # CROSS
    cross_index = cl // 2
    qq1 = [q1 for nu in range(0, n // 2, 2) for q1 in range((n // 2 + nu) * cl + cross_index, (n // 2 + nu + 1) * cl)]
    qq2 = [q2 for nu in range(0, n // 2, 2) for q2 in
           range((n // 2 + nu + 1) * cl + cross_index, (n // 2 + nu + 2) * cl)]
    cross = qm.Swap_reg(qq1, qq2, n * cl)
    mat_cross = cross.get_matrix()

    # DEFINE Ek (The operators describing the process though a generation)

    # E = np.zeros((mat_clone.shape[0]**2, mat_clone.shape[1]**2), dtype=np.float32)
    E = scipy.sparse.csr_matrix((mat_clone.shape[0] ** 2, mat_clone.shape[1] ** 2), dtype=np.float32)
    i = 0
    k = 0
    Ek_arr = []
    for Es in Es_arr:
        for Er in Er_arr:
            k += 1
            Ek = np.array(mat_cross @ mat_clone @ Er @ Es, dtype=np.float32)
            Ek_arr.append(Ek)
            np.save("Qchannel_canonical_case/E_{:d}".format(k), Ek)
            E += scipy.sparse.kron(Ek.conjugate(), Ek)
            # E += np.kron(Ek.conjugate(), Ek)
            # Ekk = Ek.conjugate()[:, np.newaxis, :, np.newaxis]
            # Ekk = Ekk[:, np.newaxis, :, np.newaxis] * Ek[np.newaxis, :, np.newaxis, :]
            # Ekk.shape = (Ek.shape[0] ** 2, Ek.shape[1] ** 2)
            # E += Ekk
        i += 1
        if show_progress:
            print("Competition: {:.2f} %".format(i / len(Es_arr) * 100))
    end = time.time()
    if show_progress:
        print("Time required: ", end - start)
        print()
        print("\n\nEIGENVALUES AND EIGENVECTORS\n OF THE QUANTUM CHANEL\n\n")
    w, vr = scipy.sparse.linalg.eigs(E, k=6, which='LM')
    A_arr = []
    for i in range(len(w)):
        if show_progress:
            if abs(w[i]) > 1e-5:
                print(i, "->", w[i])
                print()
        if abs(abs(w[i]) - 1) < 1e-1:
            A = vr[:, i].reshape(mat_clone.shape).transpose()
            A = A / A.trace()
            A_arr.append(A)
    return w, A_arr


def get_fixed_points_after_sort(u=None, show_progress=True):
    g = 5
    n = 4
    cl = 2
    pm = 1 / n / cl

    # SORT
    a = int(np.ceil(n / 2) * (n - 1))

    criteria = lambda x, y: sum(int(xi) * 2 ** (len(x) - i - 1) for i, xi in enumerate(x)) > sum(
        int(yi) * 2 ** (len(y) - i - 1) for i, yi in enumerate(y))

    # u = special_ortho_group.rvs(2 ** cl)
    # u = np.array([[0.29419042, 0.27876697, -0.66839818, 0.6236865],
    #              [0.45200546, -0.83398993, -0.28231706, -0.14299979],
    #              [0.45433315, 0.46058814, -0.27247138, -0.71217927],
    #              [-0.70903064, -0.12086497, -0.63190213, -0.28873328]])
    if u is None:
        u = np.identity(4)

    U = np.kron(u, u)
    U = np.kron(U, U)

    start = time.time()
    sort_unitary = qm.Identity(2 ** (n * cl + a))
    for stage in range(n):
        for reg in range(0, n - 1, 2):
            nu = reg + stage % 2
            if nu + 2 > n:
                continue
            reg1 = list(range(nu * cl, (nu + 1) * cl))
            reg2 = list(range((nu + 1) * cl, (nu + 2) * cl))
            ancilla = n * cl + (stage // 2) * (n - 1) + (stage % 2) * (n // 2) + reg // 2
            orac = qm.Oracle(criteria=criteria,
                             nq=n * cl + a,
                             reg1=reg1,
                             reg2=reg2,
                             ancilla=ancilla)

            cs = qm.CSwap_reg(ancilla,
                              reg1,
                              reg2,
                              n * cl + a)
            sort_unitary = cs.dot(orac).dot(sort_unitary)

    Es_arr = []
    for k in range(2 ** a):
        kbin = bin(k)[2:]
        if len(kbin) < a:
            kbin = "0" * (a - len(kbin)) + kbin

        kvalid = True
        s0us1 = [reg // 2 for reg in range(0, n - 1, 2)] + [(n // 2) + reg // 2 for reg in range(0, n - 2, 2)]
        gt = int(any(kbin[j] == "1" for j in s0us1))
        for stage in range(2, n):
            st = [(stage // 2) * (n - 1) + (stage % 2) * (n // 2) + reg // 2 for reg in range(0, n - 1 - stage % 2, 2)]
            ft = int(any(kbin[j] == "1" for j in st))
            if gt < ft:
                kvalid = False
                break

        if not kvalid:
            continue

        Ek = np.zeros((2 ** (n * cl), 2 ** (n * cl)))
        for i in range(2 ** (n * cl)):
            # for j in range(2 ** (n * cl)):
            #    Ek[i, j] = sort_unitary.get(k + i * 2 ** a, j * 2 ** a)
            for j, element in sort_unitary.row(k + i * 2 ** a):
                if j % (2 ** a) == 0:
                    Ek[i, j // (2 ** a)] = element
        if np.max(Ek) < 1e-5:
            continue

        Es_arr.append(U @ Ek @ U.transpose().conjugate())

    # RESET
    Er_arr = []
    for k in range(2 ** (n // 2 * cl)):
        Pk = np.zeros((2 ** (n // 2 * cl), 2 ** (n // 2 * cl)))
        Pk[0, k] = 1

        Ek = np.kron(np.identity(2 ** (n // 2 * cl)), Pk)
        Er_arr.append(Ek)

    # CLONE
    clone = qm.Identity(2 ** (n * cl))
    for nu in range(0, n // 2):
        s = qm.Swap_reg(range((nu + 1) * cl, (nu + 2) * cl),
                        range((nu + n // 2) * cl, (nu + 1 + n // 2) * cl),
                        n * cl)
        uc = qm.Uclone(2 ** cl)
        unu = s.dot(qm.KronExpand(2 ** (nu * cl), uc, 2 ** ((n - nu - 2) * cl))).dot(s)
        clone = clone.dot(unu)
    mat_clone = clone.get_matrix()

    # CROSS
    cross_index = cl // 2
    qq1 = [q1 for nu in range(0, n // 2, 2) for q1 in range((n // 2 + nu) * cl + cross_index, (n // 2 + nu + 1) * cl)]
    qq2 = [q2 for nu in range(0, n // 2, 2) for q2 in
           range((n // 2 + nu + 1) * cl + cross_index, (n // 2 + nu + 2) * cl)]
    cross = qm.Swap_reg(qq1, qq2, n * cl)
    mat_cross = cross.get_matrix()

    # DEFINE Ek (The operators describing the process though a generation)

    # E = np.zeros((mat_clone.shape[0]**2, mat_clone.shape[1]**2), dtype=np.float32)
    E = scipy.sparse.csr_matrix((mat_clone.shape[0] ** 2, mat_clone.shape[1] ** 2), dtype=np.float32)
    i = 0
    k = 0
    Ek_arr = []
    for Es in Es_arr:
        for Er in Er_arr:
            k += 1
            Ek = np.array(Es @ mat_cross @ mat_clone @ Er, dtype=np.float32)
            Ek_arr.append(Ek)
            #np.save("Qchannel_canonical_case/E_{:d}".format(k), Ek)
            E += scipy.sparse.kron(Ek.conjugate(), Ek)
            # E += np.kron(Ek.conjugate(), Ek)
            # Ekk = Ek.conjugate()[:, np.newaxis, :, np.newaxis]
            # Ekk = Ekk[:, np.newaxis, :, np.newaxis] * Ek[np.newaxis, :, np.newaxis, :]
            # Ekk.shape = (Ek.shape[0] ** 2, Ek.shape[1] ** 2)
            # E += Ekk
        i += 1
        if show_progress:
            print("Competition: {:.2f} %".format(i / len(Es_arr) * 100))
    end = time.time()
    if show_progress:
        print("Time required: ", end - start)
        print()

        print("\n\nEIGENVALUES AND EIGENVECTORS\n OF THE QUANTUM CHANEL\n\n")
    w, vr = scipy.sparse.linalg.eigs(E, k=6, which='LM')
    A_arr = []
    for i in range(len(w)):
        if show_progress:
            if abs(w[i]) > 1e-5:
                print(i, "->", w[i])
                print()
        if abs(abs(w[i]) - 1) < 1e-1:
            A = vr[:, i].reshape(mat_clone.shape).transpose()
            A = A / A.trace()
            A_arr.append(A)
    return w, A_arr


def get_fixed_points_after_sort_with_mutation(u=None, show_progress=True):
    g = 5
    n = 4
    cl = 2
    pm = 1 / n / cl

    # SORT
    a = int(np.ceil(n / 2) * (n - 1))

    criteria = lambda x, y: sum(int(xi) * 2 ** (len(x) - i - 1) for i, xi in enumerate(x)) > sum(
        int(yi) * 2 ** (len(y) - i - 1) for i, yi in enumerate(y))

    # u = special_ortho_group.rvs(2 ** cl)
    # u = np.array([[0.29419042, 0.27876697, -0.66839818, 0.6236865],
    #              [0.45200546, -0.83398993, -0.28231706, -0.14299979],
    #              [0.45433315, 0.46058814, -0.27247138, -0.71217927],
    #              [-0.70903064, -0.12086497, -0.63190213, -0.28873328]])
    if u is None:
        u = np.identity(4)

    U = np.kron(u, u)
    U = np.kron(U, U)

    start = time.time()
    sort_unitary = qm.Identity(2 ** (n * cl + a))
    for stage in range(n):
        for reg in range(0, n - 1, 2):
            nu = reg + stage % 2
            if nu + 2 > n:
                continue
            reg1 = list(range(nu * cl, (nu + 1) * cl))
            reg2 = list(range((nu + 1) * cl, (nu + 2) * cl))
            ancilla = n * cl + (stage // 2) * (n - 1) + (stage % 2) * (n // 2) + reg // 2
            orac = qm.Oracle(criteria=criteria,
                             nq=n * cl + a,
                             reg1=reg1,
                             reg2=reg2,
                             ancilla=ancilla)

            cs = qm.CSwap_reg(ancilla,
                              reg1,
                              reg2,
                              n * cl + a)
            sort_unitary = cs.dot(orac).dot(sort_unitary)

    Es_arr = []
    for k in range(2 ** a):
        kbin = bin(k)[2:]
        if len(kbin) < a:
            kbin = "0" * (a - len(kbin)) + kbin

        kvalid = True
        s0us1 = [reg // 2 for reg in range(0, n - 1, 2)] + [(n // 2) + reg // 2 for reg in range(0, n - 2, 2)]
        gt = int(any(kbin[j] == "1" for j in s0us1))
        for stage in range(2, n):
            st = [(stage // 2) * (n - 1) + (stage % 2) * (n // 2) + reg // 2 for reg in range(0, n - 1 - stage % 2, 2)]
            ft = int(any(kbin[j] == "1" for j in st))
            if gt < ft:
                kvalid = False
                break

        if not kvalid:
            continue

        Ek = np.zeros((2 ** (n * cl), 2 ** (n * cl)))
        for i in range(2 ** (n * cl)):
            # for j in range(2 ** (n * cl)):
            #    Ek[i, j] = sort_unitary.get(k + i * 2 ** a, j * 2 ** a)
            for j, element in sort_unitary.row(k + i * 2 ** a):
                if j % (2 ** a) == 0:
                    Ek[i, j // (2 ** a)] = element
        if np.max(Ek) < 1e-5:
            continue

        Es_arr.append(U @ Ek @ U.transpose().conjugate())

    # RESET
    Er_arr = []
    for k in range(2 ** (n // 2 * cl)):
        Pk = np.zeros((2 ** (n // 2 * cl), 2 ** (n // 2 * cl)))
        Pk[0, k] = 1

        Ek = np.kron(np.identity(2 ** (n // 2 * cl)), Pk)
        Er_arr.append(Ek)

    # CLONE
    clone = qm.Identity(2 ** (n * cl))
    for nu in range(0, n // 2):
        s = qm.Swap_reg(range((nu + 1) * cl, (nu + 2) * cl),
                        range((nu + n // 2) * cl, (nu + 1 + n // 2) * cl),
                        n * cl)
        uc = qm.Uclone(2 ** cl)
        unu = s.dot(qm.KronExpand(2 ** (nu * cl), uc, 2 ** ((n - nu - 2) * cl))).dot(s)
        clone = clone.dot(unu)
    mat_clone = clone.get_matrix()

    # CROSS
    cross_index = cl // 2
    qq1 = [q1 for nu in range(0, n // 2, 2) for q1 in range((n // 2 + nu) * cl + cross_index, (n // 2 + nu + 1) * cl)]
    qq2 = [q2 for nu in range(0, n // 2, 2) for q2 in
           range((n // 2 + nu + 1) * cl + cross_index, (n // 2 + nu + 2) * cl)]
    cross = qm.Swap_reg(qq1, qq2, n * cl)
    mat_cross = cross.get_matrix()

    # Get mutation Kraus matrices
    mutation_unitary_set = [np.identity(2),
                            np.array([[0, 1], [1, 0]]),
                            np.array([[0, -1j], [1j, 0]]),
                            np.array([[1, 0], [0, -1]])]

    Em_arr = []
    for m in itertools.product([0, 1, 2, 3], repeat=n*cl):
        Em = np.identity(1)
        for mi in m:
            Em = np.kron(Em, mutation_unitary_set[mi])
        Em_arr.append(scipy.sparse.csr_matrix(Em, dtype=np.complex64))

    # DEFINE Ek (The operators describing the process though a generation)
    raise Exception("This takes too much time.")
    # E = np.zeros((mat_clone.shape[0]**2, mat_clone.shape[1]**2), dtype=np.float32)
    E = scipy.sparse.csr_matrix((mat_clone.shape[0] ** 2, mat_clone.shape[1] ** 2), dtype=np.complex64)
    i = 0
    k = 0
    Ek_arr = []
    for Es in Es_arr:
        for Er in Er_arr:
            for Em in Em_arr:
                k += 1
                Ek = np.array(Es @ Em @ mat_cross @ mat_clone @ Er, dtype=np.complex64)
                Ek_arr.append(Ek)
                #np.save("Qchannel_canonical_case/E_{:d}".format(k), Ek)
                E += scipy.sparse.kron(Ek.conjugate(), Ek)
                # E += np.kron(Ek.conjugate(), Ek)
                # Ekk = Ek.conjugate()[:, np.newaxis, :, np.newaxis]
                # Ekk = Ekk[:, np.newaxis, :, np.newaxis] * Ek[np.newaxis, :, np.newaxis, :]
                # Ekk.shape = (Ek.shape[0] ** 2, Ek.shape[1] ** 2)
                # E += Ekk
            i += 1
            if show_progress:
                print("Competition: {:.2f} %".format(i / len(Es_arr) / len(Er_arr) * 100))
    end = time.time()
    if show_progress:
        print("Time required: ", end - start)
        print()

        print("\n\nEIGENVALUES AND EIGENVECTORS\n OF THE QUANTUM CHANEL\n\n")
    w, vr = scipy.sparse.linalg.eigs(E, k=6, which='LM')
    A_arr = []
    for i in range(len(w)):
        if show_progress:
            if abs(w[i]) > 1e-5:
                print(i, "->", w[i])
                print()
        if abs(abs(w[i]) - 1) < 1e-1:
            A = vr[:, i].reshape(mat_clone.shape).transpose()
            A = A / A.trace()
            A_arr.append(A)
    return w, A_arr


if __name__ == '__main__':
    g = 5
    n = 4
    cl = 2
    pm = 1 / n / cl

    #u = special_ortho_group.rvs(2 ** cl)
    u = np.array([[0.29419042, 0.27876697, -0.66839818, 0.6236865],
                   [0.45200546, -0.83398993, -0.28231706, -0.14299979],
                   [0.45433315, 0.46058814, -0.27247138, -0.71217927],
                   [-0.70903064, -0.12086497, -0.63190213, -0.28873328]])
    #u = np.identity(4)
    A_arr = get_fixed_points_after_sort_with_mutation(u)

    print("\n\nDENSITY MATRICES WITH w=1\n")
    fixed_point_index = 0
    for A in A_arr:
        fixed_point_index += 1
        #np.save("Qchannel_canonical_case/fixed_point_{:d}".format(fixed_point_index), A)

        pi, vi = linalg.eig(A)
        print("Eigenvalues of the density matrix:")
        print("All real (eps=1e-6)? ", np.all(abs(pi-pi.real) < 1e-6))
        print("All positive (eps=1e-6)? ", np.all(pi.real > -1e-6))
        print("All less than one (eps=1e-6)? ", np.all(pi.real < 1 + 1e-6))
        print()
        print("Trace of the density matrix:")
        print(A.trace())
        print()
        print("Diagonal terms of the density matrix:")
        print("All real (eps=1e-6)? ", np.all(abs(np.diag(A) - np.diag(A).real) < 1e-6))
        print("All positive (eps=1e-6)? ", np.all(np.diag(A).real > -1e-6))
        print("All less than one (eps=1e-6)? ", np.all(np.diag(A).real < 1 + 1e-6))
        print()

        rho_pop = qm.rho(A, dense=True)
        print("Fidelity of each state:\n")
        track_fidelity = [u[:, i] for i in range(u.shape[1])]

        fidelity_array = np.zeros((n, len(track_fidelity)))

        for reg in range(n):
            reg_state = rho_pop.partial_trace(list(range(reg * cl, (reg + 1) * cl)))
            print("reg {:d}    ".format(reg), end="")
            for i, state in enumerate(track_fidelity):
                fidelity_array[reg, i] = reg_state.fidelity(state)
                print("{:5.3f}".format(fidelity_array[reg, i]), end="\t")
            print()








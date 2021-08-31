"""
.. module:: quantum_mats.py
    :synopsis: Defines the Quantum_Matrix class and implements
    several specific matrices.
.. moduleauthor::  Ruben Ibarrondo (rubenibarrondo@gmail.com)
"""
from abc import ABC, abstractmethod
from math import log2, ceil, cos, sin, pi, sqrt
from random import random
import numpy as np
from scipy import linalg


class QuantumMatrix:

    error_tol = 1e-5

    @abstractmethod
    def get(self, i, j):
        if i >= self.shape[0] or j >= self.shape[1] or i < 0 or j < 0:
            raise IndexError(
                'Index out of bounds, obtained ({}, {}) for shape ({}, {})'.format(
                    i, j, self.shape[0], self.shape[1]))
        if type(i) != int or type(j) != int:
            raise IndexError(
                'Only int index allowed but {} and {} were obtained'.format(type(i), type(j))
            )
        pass

    @abstractmethod
    def row(self, i0):
        # [(j, value), (j, value), ...]
        if i0 >= self.shape[0] or i0 < 0:
            raise IndexError(
                'Index out of bounds, obtained ({}, :) for shape ({}, {})'.format(
                    i0, self.shape[0], self.shape[1]))
        if type(i0) != int:
            raise IndexError(
                'Only int index allowed but {} was obtained'.format(type(i0))
            )
        pass

    @abstractmethod
    def col(self, j0):
        # [(i, value), (i, value), ...]
        if j0 >= self.shape[1] or j0 < 0:
            raise IndexError(
                'Index out of bounds, obtained (:, {}) for shape ({}, {})'.format(
                    j0, self.shape[0], self.shape[1]))
        if type(j0) != int:
            raise IndexError(
                'Only int index allowed but {} was obtained'.format(type(j0))
            )
        pass

    @abstractmethod
    def dot(self, M):
        if self.shape != M.shape:
            raise IndexError(
                'Shape mismatch, obtained ({}, {}) and ({}, {})'.format(
                    self.shape[0], self.shape[1], M.shape[0], M.shape[1]))
        pass

    def print_mat(self, block=None, len_lim=32):
        if self.shape[0] > len_lim:
            raise ValueError('Too big for printing. shape = {}'.format(self.shape))
        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                print(self.get(i, j), end=' ')
                if block and j%block == block-1:
                    print(end=' ')
            print()
            if block and i%block == block - 1:
                print()

    def get_matrix(self, len_lim=2**12):
        if self.shape[0] > len_lim:
            raise ValueError('Too big for getting matrix with len_lim={}. shape = {}'.format(len_lim, self.shape))
        mat = np.zeros(self.shape)
        for i in range(self.shape[0]):
            for j, v in self.row(i):
                mat[i, j] = v
        return mat


# -------------------
#
#    Main matrices
#
# -------------------


class rho(QuantumMatrix):

    def __init__(self, data, ii=None, jj=None,  shape=None, dense=False):
        self.dense = dense
        if not dense:
            self.shape = shape
            self.store = {(i, j): d for d, i, j in zip(data, ii, jj)}
            self.nonzeros = len(self.store)
            self.nq = ceil(log2(self.shape[0]))
        if dense:
            if abs(np.trace(data) - 1) > QuantumMatrix.error_tol:
                raise ValueError("rho not normalized, tr: ", abs(np.trace(data) - 1))
            if np.any(abs(np.transpose(data.conjugate()) - data) > QuantumMatrix.error_tol):
                raise ValueError("rho not Hermitian, max dev: ", np.max(abs(np.transpose(data.conjugate()) - data)))
            if np.any(np.linalg.eigvals(data) < -QuantumMatrix.error_tol):
                raise ValueError("rho not positive define, min val: ", np.min(np.linalg.eigvals(data)))
            self.shape = data.shape
            self.store = data
            self.nonzeros = data.shape[0] * data.shape[1]
            self.nq = ceil(log2(self.shape[0]))

    def __binenc(self, i):
        ib = bin(i)[2:]
        if len(ib) < self.nq:
            ib = '0' * (self.nq - len(ib)) + ib
        return ib

    def __bindec(self, ib):
        return sum(2 ** (len(ib) - j - 1) * int(ib[j]) for j in range(len(ib)))

    def __add__(self, other):
        QuantumMatrix.dot(self, other)
        if type(other) == np.ndarray and self.dense:
            return self.store + other
        elif type(other) != rho:
            raise NotImplementedError('__add__ is only implemented for rho matrices.')

        new_rho = rho([], [], [], self.shape)
        for i, j in self.store.keys():
            new_rho.set(i, j, self.get(i, j))

        for i, j in other.store.keys():
            if new_rho.get(i, j) == 0:
                new_rho.set(i, j, other.get(i, j))
            else:
                new_rho.set(i, j, new_rho.get(i, j) + other.get(i, j))
        return new_rho

    def __sub__(self, other):
        QuantumMatrix.dot(self, other)
        if type(other) == np.ndarray and self.dense:
            return self.store - other
        elif type(other) != rho:
            raise NotImplementedError('__sub__ is only implemented for rho matrices.')

        new_rho = rho([], [], [], self.shape)
        for i, j in self.store.keys():
            new_rho.set(i, j, self.get(i, j))

        for i, j in other.store.keys():
            if new_rho.get(i, j) == 0:
                new_rho.set(i, j, -other.get(i, j))
            else:
                new_rho.set(i, j, new_rho.get(i, j) - other.get(i, j))
        return new_rho

    def __mul__(self, other):
        if (type(other) != int or type(other) != float) and self.dense:
            return self.store * other
        elif type(other) != int and type(other) != float:
            raise NotImplementedError('__mul__ is only implemented for rho with int and float, but {} was obtained'.format(type(other)))
        new_rho = rho([], [], [], self.shape)
        for i, j in self.store.keys():
            new_rho.set(i, j, other * self.get(i, j))
        return new_rho

    def __truediv__(self, other):
        if (type(other) != int or type(other) != float) and self.dense:
            return self.store / other
        elif type(other) != int and type(other) != float:
            raise NotImplementedError('__div__ is only implemented for rho with int and float, but {} was obtained'.format(type(other)))
        new_rho = rho([], [], [], self.shape)
        for i, j in self.store.keys():
            new_rho.set(i, j, self.get(i, j)/other)
        return new_rho

    def get_matrix(self, len_lim=2**12):
        if not self.dense:
            return QuantumMatrix.get_matrix(self, len_lim)
        else:
            return self.store

    @staticmethod
    def gen_rho_from_state(coefficients, state_indices, length):
        r = rho([], [], [], (length, length))

        if len(coefficients) != len(state_indices):
            raise ValueError('coefficients and state_indices must have same lenght')

        for p in range(len(state_indices)):
            for q in range(len(state_indices)):
                r.set(state_indices[p], state_indices[q], coefficients[p]*coefficients[q])
        return r

    @staticmethod
    def gen_rho_from_matrix(matrix):
        return rho(matrix, dense=True)

    @staticmethod
    def gen_random_rho(nq, dense=True, constraints={"pure": True, "separable": False}, asvector=False):
        """
        Generate a random rho state for nq qubits.
        Only dense=True and constraints["pure"] = True are supported yet.
        The random pure state generator is based on the Qiskit random_statevector function (bases in Haar measures).
        :param nq:
        :param dense:
        :param constraints:
        :param asvector:
        :return:
        """
        if not dense and not constraints.get("pure", False):
            raise ValueError("Only dense=True and constraints[\"pure\"] = True are supported yet.")
        if constraints.get("separable", False):
            raise ValueError("Constraints[\"separable\"] = True is not supported yet.")

        dim = 2**nq
        x = np.random.random(dim)
        x += x == 0
        x = -np.log(x)
        sumx = sum(x)
        phases = np.random.random(dim) * 2.0 * np.pi

        statevector = np.sqrt(x / sumx) * np.exp(1j * phases)
        if asvector and constraints.get("pure", True):
            return statevector
        else:
            mat = np.kron(statevector.reshape((statevector.shape[0], 1)), statevector.conjugate())
            return rho.gen_rho_from_matrix(mat)

    @staticmethod
    def get_xx_basis():
        pp = rho.gen_rho_from_state(np.full(4, 1 / 2), range(4), 4)
        pm = rho.gen_rho_from_state(np.full(4, 1 / 2) * np.array([1, -1, 1, -1]), range(4), 4)
        mp = rho.gen_rho_from_state(np.full(4, 1 / 2) * np.array([1, 1, -1, -1]), range(4), 4)
        mm = rho.gen_rho_from_state(np.full(4, 1 / 2) * np.array([1, -1, -1, 1]), range(4), 4)
        return pp, pm, mp, mm

    def str_qubit_state(self, block=None):
        state = ''
        for i, j in self.store:
            ib = self.__binenc(i)
            jb = self.__binenc(j)
            if not block:
                if state != '':
                    state += '\n'
                state += ib + ", " + jb + ': ' + str(self.get(i, j))
            if block:
                ibin = ''
                for k, b in enumerate(ib):
                    if k % block == 0 and k != 0:
                        ibin += ' ' + b
                    else:
                        ibin += b
                jbin = ''
                for k, b in enumerate(jb):
                    if k % block == 0 and k != 0:
                        jbin += ' ' + b
                    else:
                        jbin += b
                if state != '':
                    state += '\n'
                state += ibin + ", " + jbin + ': ' + str(self.get(i, j))
        return state

    def get(self, i, j):
        QuantumMatrix.get(self, i, j)
        if not self.dense:
            return self.store.get((i, j), 0)
        else:
            return self.store[i, j]

    def set(self, i, j, value):
        QuantumMatrix.get(self, i, j)
        # This method should also check that this matrix
        # fulfils the constrains of a state matrix.
        if not self.dense:
            if value != 0:
                self.store[(i, j)] = value
            elif (i, j) in self.store.keys():
                self.store.pop((i, j))
            self.nonzeros = len(self.store)
        else:
            self.store[i, j] = value

    def col(self, j0):
        QuantumMatrix.col(self, j0)
        if not self.dense:
            for i, j in self.store.keys():
                if j == j0:
                    yield i, self.store[(i, j)]
        else:
            return self.store[:, j0]

    def row(self, i0):
        QuantumMatrix.row(self, i0)
        if not self.dense:
            for i, j in self.store.keys():
                if i == i0:
                    yield j, self.store[(i, j)]
        else:
            return self.store[i0, :]

    def trace(self):
        if not self.dense:
            tr = 0
            for i, j in self.store.keys():
                if i == j:
                    tr += self.get(i, j)
            return tr
        else:
            return self.store.trace()

    def partial_trace(self, qubit_2_keep, len_lim=2**12):
        """ Calculate the partial trace for qubit system.
        self is casted to a matrix.
        From neversakura/parital_trace.py
        https://gist.github.com/neversakura/d6a60b4bb2990d252e9e89e5629d5553#file-parital_trace-py
        Parameters
        ----------
        qubit_2_keep: list
            Index of qubit to be kept after taking the trace
        len_lim: int
            Maximum length allowed for the state. Defaults to 2**12.
        Returns
        -------
        rho_res: rho
            Density matrix after taking partial trace, type dense
        """
        r = self.get_matrix(len_lim)
        num_qubit = int(np.log2(r.shape[0]))
        qubit_axis = [(i, num_qubit + i) for i in range(num_qubit)
                      if i not in qubit_2_keep]
        minus_factor = [(i, 2 * i) for i in range(len(qubit_axis))]
        minus_qubit_axis = [(q[0] - m[0], q[1] - m[1])
                            for q, m in zip(qubit_axis, minus_factor)]
        rho_res = np.reshape(r, [2, 2] * num_qubit)
        qubit_left = num_qubit - len(qubit_axis)
        for i, j in minus_qubit_axis:
            rho_res = np.trace(rho_res, axis1=i, axis2=j)
        if qubit_left > 1:
            rho_res = np.reshape(rho_res, [2 ** qubit_left] * 2)
        return rho(rho_res, dense=True)

    def dot(self, M):
        QuantumMatrix.dot(self, M)
        if type(M) == np.ndarray and self.dense:
            return np.dot(self.store, M)
        elif type(M) != np.ndarray and self.dense:
            raise NotImplementedError('dot product for dense state only supports np.ndarray.')
        # assuming rho has smaller nonzero number
        # add shape checking
        new_rho = rho([], [], [], self.shape)
        for i, k in self.store.keys():
            for j, v in M.row(k):
                if new_rho.get(i, j) == 0:
                    new_rho.set(i, j, v * self.get(i, k))
                else:
                    new_rho.set(i, j, new_rho.get(i, j) + v * self.get(i, k))
        return new_rho

    def set_qubit(self, q, v, measure_it='random'):
        '''
        This method can be used to set a qubit to 1 or 0.
        The process can happen in 2 ways:
            - Measure the qubit and get v
            - Measure the qubit and get !v, perform X in the qubit
        Both processes can not happen simultaneously in the same state.
        For an state that has only one value for the qubit this means nothing,
        but if there is a superposition loss of information may happen.

        If the parameter measure_it is set to 'v', the default option will be to directly measure
        the value 'v' in the register, only measuring !v if the previous state doesn't exist.
        If set to '!v' the other option will be priorized.
        If set to 'random', the intrinsic probabilities of the states will be used to choose the case.
        '''
        if v != 0 and v != 1:
            raise ValueError('Qubits can only be set to 0 or 1, but {} was obtained.'.format(v))
        if self.trace() == 0:
            raise ValueError('The rho state should have trace = 1.')

        if measure_it == 'v':
            measure_it = True
        elif measure_it == '!v':
            measure_it = False
        elif measure_it == 'random':
            r = random()
            p = 0
            for i, j in self.store.keys():
                if i == j:
                    if self.__binenc(i)[q] == str(v):
                        p += self.get(i, i)
            measure_it = r < p
        else:
            raise ValueError(
                "measure_it option can only be set to 'v', '!v' or 'random', but {} was obtained.".format(measure_it))

        if measure_it:
            new_rho = rho([], [], [], self.shape)
            for i, j in self.store.keys():
                ib = self.__binenc(i)
                jb = self.__binenc(j)
                if ib[q] == str(v) and jb[q] == str(v):
                    new_rho.set(i, j, self.get(i, j))
                else:
                    new_rho.set(i, j, 0)
            tr = new_rho.trace()
            if tr != 0:
                new_rho = new_rho / tr
                self.store = new_rho.store
                self.nonzeros = new_rho.nonzeros
            else:
                measure_it = False

        if not measure_it:
            new_rho = rho([], [], [], self.shape)
            for i, j in self.store.keys():
                ib = self.__binenc(i)
                jb = self.__binenc(j)
                if ib[q] != str(v) and jb[q] != str(v):
                    if q < self.nq-1 and q > 0:
                        inew = self.__bindec(ib[:q]+str(v)+ib[q+1:])
                        jnew = self.__bindec(jb[:q]+str(v)+jb[q+1:])
                    elif q == 0:
                        inew = self.__bindec(str(v) + ib[q + 1:])
                        jnew = self.__bindec(str(v) + jb[q + 1:])
                    elif q == self.nq-1:
                        inew = self.__bindec(ib[:q] + str(v))
                        jnew = self.__bindec(jb[:q] + str(v))

                    new_rho.set(inew, jnew, self.get(i, j))
                    new_rho.set(i, j, 0)
                else:
                    new_rho.set(i, j, 0)
            tr = new_rho.trace()
            if tr != 0:
                new_rho = new_rho / tr
                self.store = new_rho.store
                self.nonzeros = new_rho.nonzeros
            else:
                self.set_qubit(q, v, 'v')

    def projection_controlled_rotation(self, q, U0=1, U1=1, pre_projection_unitary=1, projection_method='r'):
        '''
        1-Notes
            - Check Hermitian Conjugate is working.
        :q: the qubit to apply the controlled rotation
        :U0: the unitary to be applied if 0 is obtained, default 1 for identity
        :U1: the unitary to be applied if 1 is obtained, default 1 for identity
        :pre_projection_unitary: the unitary to be applied before the projection is performed, default 1 for identity
        :projection_method: defines the way the projection is performed. If set to 'r', it computes the probability to
         measure each possibility and selects according to those weights. If set to 0 or 1, it prioritizes measuring that
         option first.
        '''

        if self.trace() == 0:
            raise ValueError('The rho state should have trace = 1.')

        if pre_projection_unitary != 1:
            new_rho = KronExpand(2 ** q, pre_projection_unitary, 2 ** (self.nq - 1 - q)).dot(
                self).dot(HermitianConjugate(KronExpand(2 ** q, pre_projection_unitary, 2 ** (self.nq - 1 - q))))
            self.store = new_rho.store
            self.nonzeros = new_rho.nonzeros

        if projection_method == 'r':
            p = 0
            for i, j in self.store.keys():
                if i == j:
                    if self.__binenc(i)[q] == '0':
                        p += self.get(i, i)
            measure_0 = random() < p
        elif projection_method == 0:
            measure_0 = True
        elif projection_method == 1:
            measure_0 = False
        else:
            raise ValueError('Parameter projection_method has to be either 0, 1 or \'r\'.')

        if measure_0:
            new_rho = rho([], [], [], self.shape)
            for i, j in self.store.keys():
                ib = self.__binenc(i)
                jb = self.__binenc(j)
                if ib[q] == '0' and jb[q] == '0':
                    new_rho.set(i, j, self.get(i, j))
                else:
                    new_rho.set(i, j, 0)
            tr = new_rho.trace()
            if tr != 0:
                new_rho = new_rho / tr
                self.store = new_rho.store
                self.nonzeros = new_rho.nonzeros
            else:
                raise ValueError('ctrl_rotation resulted in trace = 0.')

            if U0 != 1:
                new_rho = KronExpand(2 ** q, U0, 2 ** (self.nq - q - 1)).dot(self).dot(
                    HermitianConjugate(KronExpand(2 ** q, U0, 2 ** (self.nq - q - 1))))
                self.store = new_rho.store
                self.nonzeros = new_rho.nonzeros

        if not measure_0:
            new_rho = rho([], [], [], self.shape)
            for i, j in self.store.keys():
                ib = self.__binenc(i)
                jb = self.__binenc(j)
                if ib[q] == '1' and jb[q] == '1':
                    new_rho.set(i, j, self.get(i, j))
                else:
                    new_rho.set(i, j, 0)
            tr = new_rho.trace()
            if tr != 0:
                new_rho = new_rho / tr
                self.store = new_rho.store
                self.nonzeros = new_rho.nonzeros
            else:
                raise ValueError('ctrl_rotation resulted in trace = 0.')

            if U1 != 1:
                new_rho = KronExpand(2 ** q, U1, 2 ** (self.nq - q - 1)).dot(self).dot(
                    HermitianConjugate(KronExpand(2 ** q, U1, 2 ** (self.nq - q - 1))))
                self.store = new_rho.store
                self.nonzeros = new_rho.nonzeros

    def fidelity(self, other, len_lim=2**12):
        """
        Fidelity between a density matrix and a pure state/other density matrix.
        The state is casted to a np matrix.
            F(r, s) = tr^2 √(√r s √r).
        Parameters
        ----------
        other: np.ndarray, qm.rho
            Vector representation of a pure state, or qm.rho object representing other density matrix.
        len_lim: int
            Maximum length allowed for the state. Defaults to 2**12.
        Returns
        -------
        fidelity: float
            The fidelity between pure and rho
        """
        r = self.get_matrix(len_lim)
        if type(other) == np.ndarray:
            pure = other
            if abs(np.dot(pure.conjugate(), pure) - 1) > QuantumMatrix.error_tol:
                raise ValueError("pure not normalized")
            if abs(np.trace(r) - 1) > QuantumMatrix.error_tol:
                raise ValueError("rho not normalized")
            if np.any(abs(np.transpose(r.conjugate()) - r) > QuantumMatrix.error_tol):
                raise ValueError("rho not Hermitian")
            if np.any(np.linalg.eigvals(r) < -QuantumMatrix.error_tol):
                raise ValueError("rho not positive define")

            fidelity = np.dot(pure.transpose().conjugate(), np.dot(r, pure)).real
        elif type(other) == type(self):
            r2sqrt = linalg.sqrtm(other.get_matrix(len_lim))
            fidelity = np.trace(linalg.sqrtm(np.dot(r2sqrt, np.dot(r, r2sqrt)))) ** 2
        else:
            raise ValueError("other with type %s not supported." % str(type(other)))

        if fidelity.real < -QuantumMatrix.error_tol:
            raise Exception("Negative value obtained for fidelity")
        elif abs(fidelity.imag) > QuantumMatrix.error_tol:
            raise Exception("Complex value obtained for fidelity.")
        elif abs(fidelity) > 1 + QuantumMatrix.error_tol:
            raise Exception("Obtained fidelity greater than 1.")
        else:
            return min(abs(fidelity), 1)

    def get_expected_value(self, observable):
        """
        Returns the expected value of the observable.
        Both objects must be described in the same basis.

        :param observable: np.ndarray describing the matrix of the observable.
        :return:
        """
        return np.trace(np.dot(observable, self.get_matrix()))

    def entropy(self):
        """
        Computes the entropy of the density matrix.

        :return: The value of the entropy of the state.
        """
        weights = np.ma.masked_equal(np.linalg.eigvals(self.get_matrix()), 0)
        s = - np.sum(weights * np.log2(weights))
        if s.real < -QuantumMatrix.error_tol:
            raise Exception("Negative value obtained for entropy.")
        elif abs(s.imag) > QuantumMatrix.error_tol:
            raise Exception("Complex value obtained for entropy.")
        return s.real


class Uclone(QuantumMatrix):

    def __init__(self, n):
        self.n = n
        self.shape = (n * n, n * n)
        self.nonzeros = n

    def get(self, i, j):
        QuantumMatrix.get(self, i, j)
        if i // self.n == j // self.n:
            nu = i // self.n
            ip = i % self.n
            jp = j % self.n
            if jp <= self.n - nu and ip == (jp + nu):
                return 1
            elif ip == (jp + nu - self.n):
                return 1
            else:
                return 0
        else:
            return 0

    def col(self, j0):
        QuantumMatrix.col(self, j0)
        nu = j0 // self.n
        jp = j0 % self.n
        if jp <= self.n - nu - 1:
            yield nu * self.n + jp + nu, 1
        else:
            yield nu * self.n + jp + nu - self.n, 1

    def row(self, i0):
        QuantumMatrix.row(self, i0)
        nu = i0 // self.n
        ip = i0 % self.n
        if ip - nu >= 0:
            yield i0 - nu, 1
        else:
            yield i0 - nu + self.n, 1

    def dot(self, M):
        QuantumMatrix.dot(self, M)
        if type(M) == rho:
            new_rho = rho([], [], [], self.shape)
            for k, j in M.store.keys():
                for i, v in self.col(k):
                    if new_rho.get(i, j) == 0:
                        new_rho.set(i, j, v * M.get(k, j))
                    else:
                        new_rho.set(i, j, new_rho.get(i, j) + v * M.get(k, j))
            return new_rho
        else:
            return Dot(self, M)


class Swap(QuantumMatrix):

    def __init__(self, q1, q2, nq):
        self.nq = nq
        self.q1 = q1
        self.q2 = q2
        self.shape = (2 ** nq, 2 ** nq)
        self.nonzeros = 2 ** nq

    def __binenc(self, i):
        ib = bin(i)[2:]
        if len(ib) < self.nq:
            ib = '0' * (self.nq - len(ib)) + ib
        return ib

    def __bindec(self, ib):
        return sum(2 ** (len(ib) - j - 1) * int(ib[j]) for j in range(len(ib)))

    def get(self, i, j):
        QuantumMatrix.get(self, i, j)
        ibin = self.__binenc(i)
        jbin = self.__binenc(j)
        for q in range(self.nq):
            if q in [self.q1, self.q2]:
                continue
            if ibin[q] != jbin[q]:
                return 0
        if ibin[self.q1] == jbin[self.q2] and ibin[self.q2] == jbin[self.q1]:
            return 1
        else:
            return 0

    def col(self, j0):
        QuantumMatrix.col(self, j0)
        jbin = self.__binenc(j0)
        ibin = ''
        for q in range(self.nq):
            if q == self.q1:
                ibin += jbin[self.q2]
            elif q == self.q2:
                ibin += jbin[self.q1]
            else:
                ibin += jbin[q]
        yield self.__bindec(ibin), 1

    def row(self, i0):
        QuantumMatrix.row(self, i0)
        ibin = self.__binenc(i0)
        jbin = ''
        for q in range(self.nq):
            if q == self.q1:
                jbin += ibin[self.q2]
            elif q == self.q2:
                jbin += ibin[self.q1]
            else:
                jbin += ibin[q]
        yield self.__bindec(jbin), 1

    def dot(self, M):
        QuantumMatrix.dot(self, M)
        if type(M) == rho:
            new_rho = rho([], [], [], self.shape)
            for k, j in M.store.keys():
                for i, v in self.col(k):
                    if new_rho.get(i, j) == 0:
                        new_rho.set(i, j, v * M.get(k, j))
                    else:
                        new_rho.set(i, j, new_rho.get(i, j) + v * M.get(k, j))
            return new_rho
        else:
            return Dot(self, M)


class Swap_reg(QuantumMatrix):

    def __init__(self, qq1, qq2, nq):
        self.nq = nq
        self.qq1 = qq1
        self.qq2 = qq2
        self.shape = (2 ** nq, 2 ** nq)
        self.nonzeros = 2 ** nq

    def __binenc(self, i):
        ib = bin(i)[2:]
        if len(ib) < self.nq:
            ib = '0' * (self.nq - len(ib)) + ib
        return ib

    def __bindec(self, ib):
        return sum(2 ** (len(ib) - j - 1) * int(ib[j]) for j in range(len(ib)))

    def get(self, i, j):
        QuantumMatrix.get(self, i, j)
        ibin = self.__binenc(i)
        jbin = self.__binenc(j)
        for q in range(self.nq):
            if (q in self.qq1) or (q in self.qq2):
                continue
            if ibin[q] != jbin[q]:
                return 0
        if all(ibin[q1] == jbin[q2] and ibin[q2] == jbin[q1] for q1, q2 in zip(self.qq1, self.qq2)):
            return 1
        else:
            return 0

    def col(self, j0):
        QuantumMatrix.col(self, j0)
        jbin = self.__binenc(j0)
        ibin = [b for b in jbin]
        for q1, q2 in zip(self.qq1, self.qq2):
            temp = ibin[q1]
            ibin[q1] = ibin[q2]
            ibin[q2] = temp
        yield self.__bindec(''.join(ibin)), 1

    def row(self, i0):
        QuantumMatrix.row(self, i0)
        ibin = self.__binenc(i0)
        jbin = [b for b in ibin]
        for q1, q2 in zip(self.qq1, self.qq2):
            temp = jbin[q1]
            jbin[q1] = jbin[q2]
            jbin[q2] = temp
        yield self.__bindec(''.join(jbin)), 1

    def dot(self, M):
        QuantumMatrix.dot(self, M)
        if type(M) == rho:
            new_rho = rho([], [], [], self.shape)
            for k, j in M.store.keys():
                for i, v in self.col(k):
                    if new_rho.get(i, j) == 0:
                        new_rho.set(i, j, v * M.get(k, j))
                    else:
                        new_rho.set(i, j, new_rho.get(i, j) + v * M.get(k, j))
            return new_rho
        else:
            return Dot(self, M)


class CSwap(QuantumMatrix):

    def __init__(self, c, q1, q2, nq):
        self.nq = nq
        self.c = c
        self.q1 = q1
        self.q2 = q2
        self.shape = (2 ** nq, 2 ** nq)
        self.nonzeros = 2 ** nq

    def __binenc(self, i):
        ib = bin(i)[2:]
        if len(ib) < self.nq:
            ib = '0' * (self.nq - len(ib)) + ib
        return ib

    def __bindec(self, ib):
        return sum(2 ** (len(ib) - j - 1) * int(ib[j]) for j in range(len(ib)))

    def get(self, i, j):
        QuantumMatrix.get(self, i, j)
        ibin = self.__binenc(i)
        jbin = self.__binenc(j)
        if ibin[self.c] != jbin[self.c]:
            return 0
        if ibin[self.c] == '0':
            if i == j:
                return 1
            else:
                return 0
        else:
            for q in range(self.nq):
                if q in [self.q1, self.q2]:
                    continue
                if ibin[q] != jbin[q]:
                    return 0
            if ibin[self.q1] == jbin[self.q2] and ibin[self.q2] == jbin[self.q1]:
                return 1
            else:
                return 0

    def col(self, j0):
        QuantumMatrix.col(self, j0)
        jbin = self.__binenc(j0)
        if jbin[self.c] == '0':
            yield j0, 1
        else:
            ibin = ''
            for q in range(self.nq):
                if q == self.q1:
                    ibin += jbin[self.q2]
                elif q == self.q2:
                    ibin += jbin[self.q1]
                else:
                    ibin += jbin[q]
            yield self.__bindec(ibin), 1

    def row(self, i0):
        QuantumMatrix.row(self, i0)
        ibin = self.__binenc(i0)
        if ibin[self.c] == '0':
            yield i0, 1
        else:
            jbin = ''
            for q in range(self.nq):
                if q == self.q1:
                    jbin += ibin[self.q2]
                elif q == self.q2:
                    jbin += ibin[self.q1]
                else:
                    jbin += ibin[q]
            yield self.__bindec(jbin), 1

    def dot(self, M):
        QuantumMatrix.dot(self, M)
        if type(M) == rho:
            new_rho = rho([], [], [], self.shape)
            for k, j in M.store.keys():
                for i, v in self.col(k):
                    if new_rho.get(i, j) == 0:
                        new_rho.set(i, j, v * M.get(k, j))
                    else:
                        new_rho.set(i, j, new_rho.get(i, j) + v * M.get(k, j))
            return new_rho
        else:
            return Dot(self, M)


class CSwap_reg(QuantumMatrix):

    def __init__(self, c, qq1, qq2, nq):
        self.nq = nq
        self.c = c
        self.qq1 = qq1
        self.qq2 = qq2
        self.shape = (2 ** nq, 2 ** nq)
        self.nonzeros = 2 ** nq

    def __binenc(self, i):
        ib = bin(i)[2:]
        if len(ib) < self.nq:
            ib = '0' * (self.nq - len(ib)) + ib
        return ib

    def __bindec(self, ib):
        return sum(2 ** (len(ib) - j - 1) * int(ib[j]) for j in range(len(ib)))

    def get(self, i, j):
        QuantumMatrix.get(self, i, j)
        ibin = self.__binenc(i)
        jbin = self.__binenc(j)
        if ibin[self.c] != jbin[self.c]:
            return 0
        if ibin[self.c] == '0':
            if i == j:
                return 1
            else:
                return 0
        else:
            for q in range(self.nq):
                if (q in self.qq1) or (q in self.qq2):
                    continue
                if ibin[q] != jbin[q]:
                    return 0
            if all(ibin[q1] == jbin[q2] and ibin[q2] == jbin[q1] for q1, q2 in zip(self.qq1, self.qq2)):
                return 1
            else:
                return 0

    def col(self, j0):
        QuantumMatrix.col(self, j0)
        jbin = self.__binenc(j0)
        if jbin[self.c] == '0':
            yield j0, 1
        else:
            ibin = [b for b in jbin]
            for q1, q2 in zip(self.qq1, self.qq2):
                temp = ibin[q1]
                ibin[q1] = ibin[q2]
                ibin[q2] = temp
            yield self.__bindec(''.join(ibin)), 1

    def row(self, i0):
        QuantumMatrix.row(self, i0)
        ibin = self.__binenc(i0)
        if ibin[self.c] == '0':
            yield i0, 1
        else:
            jbin = [b for b in ibin]
            for q1, q2 in zip(self.qq1, self.qq2):
                temp = jbin[q1]
                jbin[q1] = jbin[q2]
                jbin[q2] = temp
            yield self.__bindec(''.join(jbin)), 1

    def dot(self, M):
        QuantumMatrix.dot(self, M)
        if type(M) == rho:
            new_rho = rho([], [], [], self.shape)
            for k, j in M.store.keys():
                for i, v in self.col(k):
                    if new_rho.get(i, j) == 0:
                        new_rho.set(i, j, v * M.get(k, j))
                    else:
                        new_rho.set(i, j, new_rho.get(i, j) + v * M.get(k, j))
            return new_rho
        else:
            return Dot(self, M)

class Identity(QuantumMatrix):

    def __init__(self, n):
        self.n = n
        self.shape = (n, n)
        self.nonzeros = n

    def get(self, i, j):
        QuantumMatrix.get(self, i, j)
        if i == j:
            return 1
        else:
            return 0

    def row(self, i0):
        QuantumMatrix.row(self, i0)
        yield i0, 1

    def col(self, j0):
        QuantumMatrix.col(self, j0)
        yield j0, 1

    def dot(self, M):
        QuantumMatrix.dot(self, M)
        return M


class Oracle(QuantumMatrix):

    def __init__(self, criteria, nq, reg1, reg2, ancilla):
        if ancilla in reg1 or ancilla in reg2:
            raise ValueError('Ancilla qubit has to be out of reg1 and reg2')
        if any(q1 in reg2 for q1 in reg1):
            raise ValueError('reg1 and reg2 can not overlap')
        self.criteria = criteria
        self.nq = nq
        self.reg1 = reg1
        self.reg2 = reg2
        self.ancilla = ancilla

        self.shape = (2**nq, 2**nq)
        self.nonzeros = 2**nq

    def __binenc(self, i):
        ib = bin(i)[2:]
        if len(ib) < self.nq:
            ib = '0' * (self.nq - len(ib)) + ib
        return ib

    def __bindec(self, ib):
        return sum(2 ** (len(ib) - j - 1) * int(ib[j]) for j in range(len(ib)))

    def get(self, i, j):
        ib = self.__binenc(i)
        jb = self.__binenc(j)

        for i, bibj in enumerate(zip(ib, jb)):
            bi, bj = bibj
            if bi != bj and i != self.ancilla:
                return 0

        qq1 = ''.join(ib[q] for q in self.reg1)
        qq2 = ''.join(ib[q] for q in self.reg2)

        a1 = jb[self.ancilla]
        a2 = ib[self.ancilla]

        doswap = self.criteria(qq1, qq2)

        if doswap:
            if a1 != a2:
                return 1
            else:
                return 0
        else:
            if a1 != a2:
                return 0
            else:
                return 1

    def row(self, i0):
        ib = self.__binenc(i0)
        qq1 = ''.join(ib[q] for q in self.reg1)
        qq2 = ''.join(ib[q] for q in self.reg2)
        a = ib[self.ancilla]
        doswap = self.criteria(qq1, qq2)

        if doswap:
            if a == '0':
                j = self.__bindec(''.join(bj if c != self.ancilla else '1' for c, bj in enumerate(ib)))
            else:
                j = self.__bindec(''.join(bj if c != self.ancilla else '0' for c, bj in enumerate(ib)))
        else:
            j = i0
        yield j, 1

    def col(self, j0):
        jb = self.__binenc(j0)
        qq1 = ''.join(jb[q] for q in self.reg1)
        qq2 = ''.join(jb[q] for q in self.reg2)
        a = jb[self.ancilla]
        doswap = self.criteria(qq1, qq2)

        if doswap:
            if a == '0':
                i = self.__bindec(''.join(bi if c != self.ancilla else '1' for c, bi in enumerate(jb)))
            else:
                i = self.__bindec(''.join(bi if c != self.ancilla else '0' for c, bi in enumerate(jb)))
        else:
            i = j0
        yield i, 1

    def dot(self, M):
        QuantumMatrix.dot(self, M)
        if type(M) == rho:
            new_rho = rho([], [], [], self.shape)
            for k, j in M.store.keys():
                for i, v in self.col(k):
                    if new_rho.get(i, j) == 0:
                        new_rho.set(i, j, v * M.get(k, j))
                    else:
                        new_rho.set(i, j, new_rho.get(i, j) + v * M.get(k, j))
            return new_rho
        else:
            return Dot(self, M)

    @staticmethod
    def get_qf_sort_oracle(nq=5, reg1=[0, 1], reg2=[2, 3], ancilla=4, uu=None, criteria=None):
        """
        Static method to construct a sort oracle for a register pair considering a quantum fitness.
        Assumes registers are consecutive.
        Parameters
        ----------
        nq: int
            Total number of registers
        reg1: list of ints
            Representation of the first register. (upper level)
        reg1: list of ints
            Representation of the second register. (lower level)
        ancilla: int
            Position of the ancilla used.
        uu: np.ndarray
            Matrix representation of the change of basis from the canonical basis to the problem basis.
            Default to None, uu = I.
        criteria: function
            Function representing the criteria for the oracle.
            Signature: criteria(x,y) -> cmp
                x: string, representing the binary representation of a state for the upper register
                y: string, representing the binary representation of a state for the lower register
                cmp: boolean, True if f(x) < f(y); False if f(x) >= f(y). Where f is the fitness in the transformed basis.
        Returns
        -------
        oracle: np.ndarray
            Matrix representing the oracle in the canonical basis
        """
        if not all(i in reg1 + reg2 for i in range(reg1[0], reg2[-1] + 1)):
            raise Exception("Only consecutive registers allowed for comparative.")

        if uu is None:
            uu = np.identity(2**(len(reg1)))

        uu = np.kron(np.kron(np.identity(2 ** reg1[0]), np.kron(uu, uu)),
                     np.identity(2 ** (nq - reg2[-1] - 1)))
        if criteria is None:
            criteria = lambda x, y: sum(int(xi) * 2 ** (len(x) - i - 1) for i, xi in enumerate(x)) > sum(
                int(yi) * 2 ** (len(y) - i - 1) for i, yi in enumerate(y))

        aux_oracle = Oracle(criteria, nq, reg1, reg2, ancilla)

        oracle = uu.dot(aux_oracle.get_matrix()).dot(np.transpose(uu.conjugate()))
        return oracle


# -------------------
#
#   Action matrices
#
# -------------------


class Transpose(QuantumMatrix):

    def __init__(self, M):
        self.M = M
        self.shape = M.shape
        self.nonzeros = M.nonzeros

    def get(self, i, j):
        return self.M.get(j, i)

    def col(self, i):
        return self.M.row(i)

    def row(self, j):
        return self.M.col(j)

    def dot(self, M):
        QuantumMatrix.dot(self, M)
        if type(M) == rho:
            new_rho = rho([], [], [], self.shape)
            for k, j in M.store.keys():
                for i, v in self.col(k):
                    if new_rho.get(i, j) == 0:
                        new_rho.set(i, j, v * M.get(k, j))
                    else:
                        new_rho.set(i, j, new_rho.get(i, j) + v * M.get(k, j))
            return new_rho
        else:
            return Dot(self, M)

class HermitianConjugate(QuantumMatrix):

    def __init__(self, M):
        self.M = M
        self.shape = M.shape
        self.nonzeros = M.nonzeros

    def get(self, i, j):
        return self.M.get(j, i).conjugate()

    def col(self, i):
        for j, v in self.M.row(i):
            yield j, v.conjugate()

    def row(self, j):
        for i, v in self.M.col(j):
            yield i, v.conjugate()

    def dot(self, M):
        QuantumMatrix.dot(self, M)
        if type(M) == rho:
            new_rho = rho([], [], [], self.shape)
            for k, j in M.store.keys():
                for i, v in self.col(k):
                    if new_rho.get(i, j) == 0:
                        new_rho.set(i, j, v * M.get(k, j))
                    else:
                        new_rho.set(i, j, new_rho.get(i, j) + v * M.get(k, j))
            return new_rho
        else:
            return Dot(self, M)


class Dot(QuantumMatrix):

    def __init__(self, A, B):
        if A.shape != B.shape:
            raise IndexError(
                'Shape mismatch, obtained ({}, {}) and ({}, {})'.format(
                    A.shape[0], A.shape[1], B.shape[0], B.shape[1]))
        self.A = A
        self.B = B
        self.shape = A.shape
        self.nonzeros = A.shape[0] * A.shape[0] # This is a maximum

    def get(self, i, j):
        return sum(va * self.B.get(k, j) for k, va in self.A.row(i))

    def col(self, j0):
        QuantumMatrix.col(self, j0)
        ii = {}
        for k, vb in self.B.col(j0):
            for i, va in self.A.col(k):
                ii[i] = va*vb + ii.get(i, 0)
        return ii.items()


    def row(self, i0):
        QuantumMatrix.row(self, i0)
        jj = {}
        for k, va in self.A.row(i0):
            for j, vb in self.B.row(k):
                jj[j] = va*vb + jj.get(j, 0)
        return jj.items()

    def dot(self, M):
        QuantumMatrix.dot(self, M)
        if type(M) == rho:
            new_rho = rho([], [], [], self.shape)
            for k, j in M.store.keys():
                for i, v in self.col(k):
                    if new_rho.get(i, j) == 0:
                        new_rho.set(i, j, v * M.get(k, j))
                    else:
                        new_rho.set(i, j, new_rho.get(i, j) + v * M.get(k, j))
            return new_rho
        else:
            return Dot(self, M)


class Kronecker(QuantumMatrix):

    def __init__(self, A, B):
        self.A = A
        self.na = A.shape[0]
        self.B = B
        self.nb = B.shape[0]
        self.shape = (self.na*self.nb, self.na*self.nb)
        self.nonzeros = A.nonzeros * B.nonzeros

    def get(self, i, j):
        return self.A.get(i//self.nb, j//self.nb) * self.B.get(i%self.nb, j%self.nb)

    def col(self, j0):
        QuantumMatrix.col(self, j0)
        ja = j0 // self.nb
        jb = j0 % self.nb

        for ia, va in self.A.col(ja):
            for ib, vb in self.B.col(jb):
                yield ia * self.nb + ib, va * vb

    def row(self, i0):
        QuantumMatrix.row(self, i0)
        ia = i0 // self.nb
        ib = i0 % self.nb

        for ja, va in self.A.row(ia):
            for jb, vb in self.B.row(ib):
                yield ja*self.nb + jb, va*vb

    def dot(self, M):
        QuantumMatrix.dot(self, M)
        if type(M) == rho:
            new_rho = rho([], [], [], self.shape)
            for k, j in M.store.keys():
                for i, v in self.col(k):
                    if new_rho.get(i, j) == 0:
                        new_rho.set(i, j, v * M.get(k, j))
                    else:
                        new_rho.set(i, j, new_rho.get(i, j) + v * M.get(k, j))
            return new_rho
        else:
            return Dot(self, M)


class KronExpand(QuantumMatrix):

    def __init__(self, n1, U, n2):
        self.u = U
        self.n = U.shape[0]
        self.n1 = n1
        self.n2 = n2
        self.shape = (self.n1 * self.n * self.n2, self.n1 * self.n * self.n2)
        self.nonzeros = self.u.nonzeros * self.n1 * self.n2

    def get(self, i, j):
        if i//(self.n*self.n2) == j//(self.n*self.n2):
            if i%(self.n*self.n2)%self.n2 == j%(self.n*self.n2)%self.n2:
                return self.u.get((i%(self.n*self.n2))//self.n2, (j%(self.n*self.n2))//self.n2)
            else:
                return 0
        else:
            return 0

    def col(self, j0):
        QuantumMatrix.col(self, j0)
        nu = j0 // (self.n*self.n2)
        pr = (j0 % (self.n*self.n2)) % self.n2
        for i, v in self.u.col((j0%(self.n*self.n2))//self.n2):
            yield i*self.n2 + pr + nu*self.n*self.n2, v

    def row(self, i0):
        QuantumMatrix.row(self, i0)
        nu = i0 // (self.n * self.n2)
        pr = (i0 % (self.n * self.n2)) % self.n2
        for j, v in self.u.row((i0%(self.n*self.n2))//self.n2):
            yield j * self.n2 + pr + nu * self.n * self.n2, v

    def dot(self, M):
        QuantumMatrix.dot(self, M)
        if type(M) == rho:
            new_rho = rho([], [], [], self.shape)
            for k, j in M.store.keys():
                for i, v in self.col(k):
                    if new_rho.get(i, j) == 0:
                        new_rho.set(i, j, v * M.get(k, j))
                    else:
                        new_rho.set(i, j, new_rho.get(i, j) + v * M.get(k, j))
            return new_rho
        else:
            return Dot(self, M)


def test_cloning_routine(N=8, cl=4):
    clone = Identity(2 ** (N * cl))
    for nu in range(0, N // 2):
        s = Swap_reg(range((nu + 1) * cl, (nu + 2) * cl),
                     range((nu + N // 2) * cl, (nu + 1 + N // 2) * cl),
                     N * cl)
        u = Uclone(2 ** cl)
        unu = s.dot(KronExpand(2 ** (nu * cl), u, 2 ** ((N - nu - 2) * cl))).dot(s)
        clone = clone.dot(unu)

    inputs = [int(i * 2 ** (cl * N / 2)) for i in range(int(2 ** (cl * (N / 2 - 1))))]
    outputs = [int(i * 2 ** (cl * N / 2) + i) for i in range(int(2 ** (cl * (N / 2 - 1))))]

    for inp, ou in zip(inputs, outputs):
        r = rho([1], [inp], [inp], (2 ** (N * cl), 2 ** (N * cl)))

        # Here Transpose <-> HermitianConjugate
        r2 = clone.dot(r).dot(Transpose(clone))
        if r2.get(ou, ou) != 1:
            print('ALERT at ', inp, ou)


if __name__ == '__main__':
    theta = pi / 4
    H = rho([cos(theta), -cos(theta), sin(theta), sin(theta)], [0, 1, 0, 1], [0, 1, 1, 0],
               (2, 2))
    X = rho([1, 1], [0, 1], [1, 0], (2, 2))
    for i in range(4):
        r = rho.gen_rho_from_state([1/sqrt(2), 1/sqrt(2)], [0, 1], 2**2)
        print(r.str_qubit_state())
        r.projection_controlled_rotation(0, 1, X, H)
        print('->', r.str_qubit_state())
        print(r.trace())
        print()
    import numpy as np
    r2 = rho.gen_rho_from_matrix(np.array([[1, 1], [1, 1]]))
    print(r2)

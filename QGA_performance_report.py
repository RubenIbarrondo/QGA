"""
.. module:: QGA_QF_performance_report.py
    :synopsis: Some functions to analyse and plot the performance of QGA
    with quantum fitness.
.. moduleauthor::  Ruben Ibarrondo (rubenibarrondo@gmail.com)
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from QGA_BCQO_sim import *
import sys


def get_overall_success_probability(rho, n, cl, success_state):
    """
    Computes:
        1 - tr[rho @ (I - |õ><õ|)^n]
    where rho is the state of the population and |õ> is the success
    state described by success_state.
    This is a measure of the probability that at least in one of the registers/individuals
    the success state will be measured.

    :param rho: qm.rho. Density matrix of the population (n * cl qubits).
    :param n: int. Number of individuals.
    :param cl: int. Chromosome length.
    :param success_state: np.array. Pure state describing the success state.
    :return:
    """
    mat = np.identity(2 ** cl) - np.outer(success_state, success_state)
    for r in range(n-1):
        mat = np.kron(mat, np.identity(2 ** cl) - np.outer(success_state, success_state))
    prob = 1 - np.trace(np.dot(rho.get_matrix(), mat))
    if prob.real < -1e-6:
        raise Exception("Negative value obtained for probability.")
    elif abs(prob.imag) > 1e-6:
        raise Exception("Complex value obtained for probability.")
    elif abs(prob) > 1 + 1e-6:
        raise Exception("Obtained probability greater than 1.")
    else:
        return min(abs(prob), 1)


def get_entropy_gap(rho, n, cl):
    """
    Computes:
        sum_r S_r - S,
    where S_r is the entropy of the reduced density matrix of each register and S
    is the entropy of the state of the population.
    This gap measures the degree of entanglement, its should be between 0 and n*cl*ln(2).


    :param rho: qm.rho. Density matrix of the population (n * cl qubits).
    :param n: int. Number of individuals.
    :param cl: int. Chromosome length.
    :return:
    """
    entropy_gap = - rho.entropy()
    for r in range(n-1):
        r = rho.partial_trace(list(range(r*cl, r*cl+cl)))
        entropy_gap += r.entropy()
    return entropy_gap


def get_concordance_product(fidelity_matrix):
    """
    Computes the concordance of a population described by the fidelity matrix.
    The fidelity matrix contains the fidelity of each register with each state
    of the problem basis.
    For each register we compute the sorted fidelity vector, sorted(fidelity_matrix[register, :], reverse=True).
    Then the concordance for each register is
            fidelity_matrix[register, :] @ sorted(fidelity_matrix[register, :], reverse=True) / norm(fidelity_matrix[register, :])**2
    This function returns the average value of the concordance in each register.

    :param fidelity_matrix:
    :return: concordance product
    """
    concordance_av = 0
    for row in fidelity_matrix:
        concordance_av += row @ sorted(row, reverse=True) / np.linalg.norm(row)**2
    concordance_av /= fidelity_matrix.shape[0]
    return concordance_av


def parse_fidelity_track(filepath, version=0):
    # adaptar para que pueda leer arrais escritos en varias líneas...
    states = np.zeros((4, 4))
    #fidelity_track = np.zeros(shape=(10, 5, 4, 4))

    if version == 0:
        with open(filepath) as file:
            fidelity_track_str = "np."
            for line_index, line in enumerate(file):
                if line_index <= 8 and line_index % 2 == 0 and line_index != 0:
                    states[line_index//2-1, :] = eval("np."+line)
                elif line_index > 9:
                    fidelity_track_str += line
            fidelity_track = eval(fidelity_track_str)
    else:
        raise NotImplementedError("version != 0 not supported.")
    return states, fidelity_track


def get_basis_sparseness(basis):
    """
    Asigns max value to |u[i,j]|**2 ~ 1/4 and minimum to |u[i,j_i]|**2 ~ 1, |u[i,j]|**2 ~ 0 if j!=j_i.
    :param basis:
    :return:
    """
    d = len(basis)
    s = 0
    for ui in basis:
        for uij in ui:
            if uij != 0:
                s += abs(uij) ** 2 * np.log(abs(uij) ** 2)
    if - s / d > -1e-6:
        return abs(s/d)
    else:
        raise Exception("Negative sparsity obtained, s = %f" % (-s/d))


def get_basis_cosine_similarity(basis_a, basis_b=None):
    """
    Cosine similarity between two vectors is computed with their dot product:
        sim(v, u) = abs(v @ u) / norm(v) / norm(u),
    or if both vectors are normalized,
        sim(v, u) = abs(v @ u).
    We define the cosine similarity for two orthonormal basis, A and B, as the minimum similarity from the set of
    maximum similarities of each vector in A with each vector in B. That is,
        sim(A, B) = min_i(max_j(abs(ai @ bi))),
    where ai in A and bi in B, each has n vectors.

    The highest value is 1, for identical basis, and the lowest is 1/sqrt(n), for completely unrelated basis.

    :param basis_a:
    :param basis_b:
    :return:
    """

    if basis_b is None:
        basis_b = np.identity(basis_a.shape)

    similarity = np.min(np.max(abs(np.transpose(np.conjugate(basis_a)) @ basis_b), axis=0))
    return similarity


def get_success_criteria(fidelity_track_array=None, reference_stage=0, g=4):
    """
    Methods to describe the success of a QGA with QF.
    :param fidelity_track_array:
    :param reference_stage: default to 0 (pre sort). Other, 1=sorted, 2=cleared, 3=cloned, 4=mixed, 5=mutated.
    :return:
    """

    success_criteria = {"probability of the best (combined)": 0,
                        "probability of the best (reg1)": 0,
                        "probability of the best (reg2)": 0,
                        "probability of the best (reg3)": 0,
                        "probability of the best (reg4)": 0,
                        "average position": 0,
                        "concordance within register": 0,
                        "concordance between registers": 0,
                        "distinguish-ability overall": 0,
                        "distinguish-ability elitist": 0,
                        "distinguish-ability overall reg1": 0,
                        "distinguish-ability elitist reg1": 0,
                        "replicability": 0,
                        "convergence stability": 0}

    trials = len(fidelity_track_array)
    reg_num = 4
    state_num = 4
    stage_num = 5

    av_f = np.zeros(shape=(stage_num, reg_num, state_num))
    av_f2 = np.zeros(shape=(stage_num, reg_num, state_num))

    for fidelity_track in fidelity_track_array:

        success_criteria["distinguish-ability overall reg1"] += fidelity_track[g, reference_stage, 0, 0] - \
                                                           fidelity_track[g, reference_stage, 0, state_num - 1]
        success_criteria["distinguish-ability elitist reg1"] += fidelity_track[g, reference_stage, 0, 0] - \
                                                           fidelity_track[g, reference_stage, 0, 1]
        not_prob = 1
        for register in range(reg_num):
            not_prob *= 1 - fidelity_track[g, reference_stage, register, 0]

            success_criteria["probability of the best (reg%d)" % (register+1)] += fidelity_track[g, reference_stage, register, 0]

            success_criteria["distinguish-ability overall"] += fidelity_track[g, reference_stage, register, 0] - fidelity_track[g, reference_stage, register, state_num-1]
            success_criteria["distinguish-ability elitist"] += fidelity_track[g, reference_stage, register, 0] - fidelity_track[g, reference_stage, register, 1]

            for register_b in range(register + 1, reg_num):
                success_criteria["concordance between registers"] += fidelity_track[g, reference_stage, register, 0] >= \
                                                                     fidelity_track[g, reference_stage, register_b, 0]
            for state in range(state_num):
                success_criteria["average position"] += state * fidelity_track[g, reference_stage, register, state]

                for stage in range(stage_num):
                    av_f[stage, register, state] += fidelity_track[g, stage, register, state]
                    av_f2[stage, register, state] += fidelity_track[g, stage, register, state] ** 2
                    success_criteria["convergence stability"] += - abs(
                        fidelity_track[g,stage, register, state] - fidelity_track[g - 1, stage, register, state])

                for state_b in range(state+1, state_num):
                    success_criteria["concordance within register"] += fidelity_track[g, reference_stage, register, state] >=\
                                                                       fidelity_track[g, reference_stage, register, state_b]
        success_criteria["probability of the best (combined)"] += 1 - not_prob

    success_criteria["probability of the best (combined)"] /= trials
    success_criteria["probability of the best (reg1)"] /= trials
    success_criteria["probability of the best (reg2)"] /= trials
    success_criteria["probability of the best (reg3)"] /= trials
    success_criteria["probability of the best (reg4)"] /= trials

    success_criteria["distinguish-ability overall"] /= trials * reg_num
    success_criteria["distinguish-ability elitist"] /= trials * reg_num
    success_criteria["distinguish-ability overall reg1"] /= trials
    success_criteria["distinguish-ability elitist reg1"] /= trials
    success_criteria["average position"] /= trials * reg_num * state_num
    success_criteria["concordance within register"] /= trials * reg_num * (np.math.factorial(state_num-1))
    success_criteria["concordance between registers"] /= trials * (np.math.factorial(reg_num-1))

    success_criteria["convergence stability"] /= trials * reg_num * state_num * stage_num
    success_criteria["convergence stability"] += 1

    df = 0
    for register in range(reg_num):
        for state in range(state_num):
            for stage in range(stage_num):
                df += av_f2[stage, register, state] / trials - (av_f[stage, register, state] / trials) ** 2
    df /= reg_num * state_num * stage_num
    if df >= 1e-6:
        success_criteria["replicability"] += 1/df
    else:
        success_criteria["replicability"] += 1e6

    return success_criteria


if __name__ == '__main__':
    if len(sys.argv) == 2:
        bigdir = sys.argv[1]
        dp_root = bigdir + '/' + "QGA_QF_test_"
        ref_stage = 1  # population after sort
        g = 4
    elif len(sys.argv) == 3:
        bigdir = sys.argv[1]
        dp_root = bigdir + '/' + "QGA_QF_test_"
        ref_stage = int(sys.argv[2])
        g = 4
    elif len(sys.argv) == 4:
        bigdir = sys.argv[1]
        dp_root = bigdir + '/' + "QGA_QF_test_"
        ref_stage = int(sys.argv[2])
        g = int(sys.argv[3])
    elif len(sys.argv) == 5:
        bigdir = sys.argv[1]
        dp_root = bigdir + '/' + "QGA_UQCM_test_"
        ref_stage = int(sys.argv[2])
        g = int(sys.argv[3])
    else:
        print(sys.argv)
        bigdir = 'QGA_QF_run_01'
        dp_root = 'QGA_QF_run_01/QGA_QF_test_'

    states, tracked_fidelity = parse_fidelity_track(dp_root+"001/fidelity_tracks_000")
    tf_dict = {}
    sc = get_success_criteria([tracked_fidelity])
    for key in sc.keys():
        tf_dict[key] = []
    tf_dict["Sparsity of the basis"] = []
    tf_dict["Sparsity of e0"] = []
    tf_dict["Sparsity of e1"] = []
    tf_dict["Sparsity of e2"] = []
    tf_dict["Sparsity of e3"] = []
    tf_dict["basis"] = []

    dir_index = 0
    while os.path.exists(dp_root + ("%03d" % (dir_index+1))):
        dir_index += 1
        directory = dp_root + ("%03d" % dir_index)

        fidelity_track_array = []
        trial_index = 0
        while os.path.isfile(directory + "/fidelity_tracks_%03d" % trial_index):
            states, tracked_fidelity = parse_fidelity_track(directory + "/fidelity_tracks_%03d" % trial_index)
            fidelity_track_array.append(tracked_fidelity)
            trial_index += 1

        tf_dict["basis"].append(states)
        sc = get_success_criteria(fidelity_track_array, reference_stage=ref_stage, g=g)
        for key, element in sc.items():
            tf_dict[key].append(element)
        tf_dict["Sparsity of the basis"].append(get_basis_sparseness(states))
        for i in range(4):
            tf_dict["Sparsity of e"+str(i)].append(get_basis_sparseness([states[i, :]]))

        print("Case %03d" % dir_index)
        print("Sparsity of the basis: %.3f" % get_basis_sparseness(states))
        print()
        for key, value in sc.items():
            print(key, value, sep=":\t")
        print("\n"+"-"*32+"\n")

    df = pd.DataFrame.from_dict(tf_dict)
    print(df)
    if ref_stage == 1:
        df.to_csv(bigdir + '/' + bigdir + "_dataframe")
        print("Saved in path:\n" + bigdir+'/'+bigdir+'_dataframe')
    else:
        df.to_csv(bigdir + '/' + bigdir + "_dataframe"+"_%d" % ref_stage)


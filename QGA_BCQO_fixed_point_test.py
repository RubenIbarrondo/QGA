import quantum_mats as qm
import numpy as np
import scipy.linalg as linalg
import time
import pandas as pd
from functools import partial
from QGA_BCQO_quantum_channels import *
import QGA_performance_report as pr
import os


def get_fixed_point_stats(save_dir):
    number_of_eigenvectors_crossover = []
    eigval2_crossover = []
    correct_density_matrix_crossover = []
    av_norm_distance_crossover = []
    sim_under_std_crossover = []
    sim_under_3_crossover = []

    number_of_eigenvectors_sort = []
    eigval2_sort = []
    correct_density_matrix_sort = []
    av_norm_distance_sort = []
    sim_under_std_sort = []
    sim_under_3_sort = []

    stats_dict = dict()

    stats_dict['number_of_eigenvectors_crossover'] = number_of_eigenvectors_crossover
    stats_dict['correct_density_matrix_crossover'] = correct_density_matrix_crossover
    stats_dict['eigval2_crossover'] = eigval2_crossover
    stats_dict['av_norm_distance_crossover'] = av_norm_distance_crossover
    stats_dict['sim_under_std_crossover'] = sim_under_std_crossover
    stats_dict['sim_under_3_crossover'] = sim_under_3_crossover

    stats_dict['number_of_eigenvectors_sort'] = number_of_eigenvectors_sort
    stats_dict['eigval2_sort'] = eigval2_sort
    stats_dict['correct_density_matrix_sort'] = correct_density_matrix_sort
    stats_dict['av_norm_distance_sort'] = av_norm_distance_sort
    stats_dict['sim_under_std_sort'] = sim_under_std_sort
    stats_dict['sim_under_3_sort'] = sim_under_3_sort

    file_index = 1
    while os.path.exists(save_dir + '/fixed_point_notes_test_{:03d}'.format(file_index)):
        incrossover = True
        save_multiplicity = False
        save_eigval2 = False
        extend_eigvals = False

        for element in stats_dict.values():
            element.append(None)

        with open(save_dir + '/fixed_point_notes_test_{:03d}'.format(file_index)) as notes:
            for line in notes:

                if "Multiplicity of eigenvalue 1:" in line:
                    save_multiplicity = True
                    continue
                if save_multiplicity:
                    if incrossover:
                        number_of_eigenvectors_crossover[-1] = int(line)
                    else:
                        number_of_eigenvectors_sort[-1] = int(line)
                    save_multiplicity = False

                if "First eigenvalues:" in line:
                    save_eigval2 = True
                    continue
                if save_eigval2:
                    if extend_eigvals:
                        eig2_str += line.strip()
                        if eig2_str[-1] == ")":
                            extend_eigvals = False
                    else:
                        eig2_str = line.strip()
                        extend_eigvals = True

                    if not extend_eigvals:
                        if "dtype" in eig2_str:
                            eig2 = [complex("".join(v.split())) for v in eig2_str[7:-2].split(",")[:-2]]
                            #eig2 += [complex("".join(v[:-1].split())) for v in eig2_str[7:-2].split(",")[-2]]
                        else:
                            eig2 = [complex("".join(v.split())) for v in eig2_str[7:-2].split(",")]
                        #assert all([e.imag < 1e-6 for e in eig2]), "Complex eigenvalues"
                        eig2 = sorted([e.real for e in eig2], reverse=True)
                        if abs(eig2[0]-1) > 0.001:
                            print(eig2)
                        if abs(eig2[1]-1) < 0.001:
                            print(eig2)
                        eig2 = eig2[1]

                        if incrossover:
                            eigval2_crossover[-1] = eig2
                        else:
                            eigval2_sort[-1] = eig2
                        save_eigval2 = False

                if "Correct density matrix:" in line:
                    if incrossover:
                        correct_density_matrix_crossover[-1] = ('True' in line)
                    else:
                        correct_density_matrix_sort[-1] = ('True' in line)

                elif "Average normalized distance:" in line:
                    if incrossover:
                        av_norm_distance_crossover[-1] = float(line[-5:])
                    else:
                        av_norm_distance_sort[-1] = float(line[-5:])

                elif "Similar values under std:" in line:
                    if incrossover:
                        sim_under_std_crossover[-1] = float(line[-5:])
                    else:
                        sim_under_std_sort[-1] = float(line[-5:])

                elif "Fidelity of each state and register (from fixed point):" in line:
                    if incrossover:
                        sim_under_3_crossover[-1] = None
                    else:
                        sim_under_3_sort[-1] = None

                elif "SORT" in line:
                    incrossover = False

        file_index += 1

    df = pd.DataFrame.from_dict(stats_dict)
    df.to_csv(save_dir + '/' + "Fixed_point_stats_dataframe")
    return df


def get_fixed_points(dp_root='QGA_BCQO_run_01/QGA_QF_test_'):
    n = 4
    cl = 2

    save_dir = 'Fixed_points_run_02'

    dir_index = 0
    while os.path.exists(dp_root + ("%03d" % (dir_index+1))):
        print("Test:", dir_index, end=' ')
        start = time.time()

        dir_index += 1
        directory = dp_root + ("%03d" % dir_index)

        fidelity_track_array = []
        trial_index = 0
        while os.path.isfile(directory + "/fidelity_tracks_%03d" % trial_index):
            states, tracked_fidelity = pr.parse_fidelity_track(directory + "/fidelity_tracks_%03d" % trial_index)
            fidelity_track_array.append(tracked_fidelity)
            trial_index += 1

        fidelity_track_array = np.array(fidelity_track_array)  # test, generation, stage, register, state
        u = states.transpose()

        w_crossover, a_arr_crossover = get_fixed_points_after_crossover(u, show_progress=False)
        w_sort, a_arr_sort = get_fixed_points_after_sort(u, show_progress=False)

        fidelity_crossover_av = np.mean(fidelity_track_array[:, -1, -1, :, :], axis=0)
        fidelity_crossover_sd = np.std(fidelity_track_array[:, -1, -1, :, :], axis=0)
        fidelity_sort_av = np.mean(fidelity_track_array[:, -1, 1, :, :], axis=0)
        fidelity_sort_sd = np.std(fidelity_track_array[:, -1, 1, :, :], axis=0)

        if len(a_arr_crossover) == 1:
            np.save(save_dir+'/fixed_point_after_crossover_test_{:03d}'.format(dir_index), a_arr_crossover[0])
        else:
            np.savez(save_dir+'/fixed_point_after_crossover_test_{:03d}'.format(dir_index), a_arr_crossover)

        if len(a_arr_sort) == 1:
            np.save(save_dir+'/fixed_point_after_sort_test_{:03d}'.format(dir_index), a_arr_sort[0])
        else:
            np.savez(save_dir+'/fixed_point_after_sort_test_{:03d}'.format(dir_index), a_arr_sort)

        with open(save_dir + '/fixed_point_notes_test_{:03d}'.format(dir_index), 'w') as notes:
            myprint = partial(print, file=notes, end="\n\n", sep='')
            myprint("Run: %d" % 2)
            myprint("Test: %d" % dir_index)
            myprint("Problem unitary:\n", repr(u))

            myprint("-" * 32)
            myprint("    NOTES ABOUT FIXED POINTS\n        AFTER CROSSOVER")
            myprint("-" * 32)

            myprint("Multiplicity of eigenvalue 1:\n{:d}".format(len(a_arr_crossover)))
            myprint("First eigenvalues:\n", repr(w_crossover))

            if len(a_arr_crossover) == 1:
                try:
                    rho_crossover = qm.rho(a_arr_crossover[0], dense=True)
                    myprint("Correct density matrix: True")

                    myprint("Fidelity of each state and register (from fixed point):")
                    track_fidelity = [u[:, i] for i in range(u.shape[1])]
                    fidelity_crossover_fp = np.zeros((n, len(track_fidelity)))
                    table_str = ' ' * 9 + (' ' * 7).join(['e0', 'e1', 'e2', 'e3'])
                    for reg in range(n):
                        table_str += '\n'
                        reg_state = rho_crossover.partial_trace(list(range(reg * cl, (reg + 1) * cl)))
                        reg_line = "reg {:d}    ".format(reg)
                        for i, state in enumerate(track_fidelity):
                            fidelity_crossover_fp[reg, i] = reg_state.fidelity(state)
                            reg_line += "{:5.3f}    ".format(fidelity_crossover_fp[reg, i])
                        table_str += reg_line
                    myprint(table_str)

                    myprint("Fidelity of each state and register (from numerical simulation):")
                    table_str = ' ' * 9 + (' ' * 7).join(['e0', 'e1', 'e2', 'e3'])
                    for reg in range(n):
                        table_str += '\n'
                        reg_line = "reg {:d}    ".format(reg)
                        for i, state in enumerate(track_fidelity):
                            reg_line += "{:5.3f}    ".format(fidelity_crossover_av[reg, i])
                        table_str += reg_line
                    myprint(table_str)

                    myprint("Average normalized distance: {:4.3f}".format(
                        np.mean(abs(fidelity_crossover_fp - fidelity_crossover_av) / (fidelity_crossover_sd+1e-6))))
                    myprint("Similar values under std: {:4.3f}".format(
                        np.mean(abs(fidelity_crossover_fp - fidelity_crossover_av) <= (fidelity_crossover_sd+1e-6))))

                except ValueError as exc:
                    myprint("Correct density matrix: False")
                    myprint("Error code:\n", str(exc))

            myprint("-" * 32)
            myprint("    NOTES ABOUT FIXED POINTS\n           AFTER SORT")
            myprint("-" * 32)

            myprint("Multiplicity of eigenvalue 1:\n{:d}".format(len(a_arr_sort)))
            myprint("First eigenvalues:\n", repr(w_sort))

            if len(a_arr_sort) == 1:
                try:
                    rho_sort = qm.rho(a_arr_sort[0], dense=True)
                    myprint("Correct density matrix: True")

                    myprint("Fidelity of each state and register (from fixed point):")
                    track_fidelity = [u[:, i] for i in range(u.shape[1])]
                    fidelity_sort_fp = np.zeros((n, len(track_fidelity)))
                    table_str = ' ' * 9 + (' ' * 7).join(['e0', 'e1', 'e2', 'e3'])
                    for reg in range(n):
                        table_str += '\n'
                        reg_state = rho_sort.partial_trace(list(range(reg * cl, (reg + 1) * cl)))
                        reg_line = "reg {:d}    ".format(reg)
                        for i, state in enumerate(track_fidelity):
                            fidelity_sort_fp[reg, i] = reg_state.fidelity(state)
                            reg_line += "{:5.3f}    ".format(fidelity_sort_fp[reg, i])
                        table_str += reg_line
                    myprint(table_str)

                    myprint("Fidelity of each state and register (from numerical simulation):")
                    table_str = ' ' * 9 + (' ' * 7).join(['e0', 'e1', 'e2', 'e3'])
                    for reg in range(n):
                        table_str += '\n'
                        reg_line = "reg {:d}    ".format(reg)
                        for i, state in enumerate(track_fidelity):
                            reg_line += "{:5.3f}    ".format(fidelity_sort_av[reg, i])
                        table_str += reg_line
                    myprint(table_str)

                    myprint("Average normalized distance: {:4.3f}".format(
                        np.mean(abs(fidelity_sort_fp - fidelity_sort_av) / (fidelity_sort_sd+1e-6))))
                    myprint("Similar values under std: {:4.3f}".format(
                        np.mean(abs(fidelity_sort_fp - fidelity_sort_av) <= (fidelity_sort_sd+1e-6))))

                except ValueError as exc:
                    myprint("Correct density matrix: False")
                    myprint("Error code:\n", str(exc))

        print(time.time() - start)


if __name__ == '__main__':
    run_num = 1
    save_dir = 'Fixed_points_BCQO_run_{:d}'.format(run_num)
    stats_df = get_fixed_point_stats(save_dir)
    print(stats_df)

"""
.. module:: QGA_performance_report_simbasis.py
    :synopsis: Some functions to analyse and plot the performance of QGA,
    for the case where similar basis functions are being analysed.
.. moduleauthor::  Ruben Ibarrondo (rubenibarrondo@gmail.com)
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from QGA_QF import *
import QGA_QF_performance_report as pr
import sys

if __name__ == '__main__':
    if len(sys.argv) == 2:
        bigdir = sys.argv[1]
        dp_root = bigdir + '/' + 'state_case_{:03d}/base_case_{:03}'
        ref_stage = 1  # population after sort
        g = 4

    assert False, "Da error al leer los estados. No todos ocupan una línea."

    states, tracked_fidelity = pr.parse_fidelity_track(dp_root.format(0, 0) + "/fidelity_tracks_000")
    tf_dict = {}
    sc = pr.get_success_criteria([tracked_fidelity])
    for key in sc.keys():
        tf_dict[key] = []
    tf_dict["Sparsity of the basis"] = []
    tf_dict["Sparsity of e0"] = []
    tf_dict["Sparsity of e1"] = []
    tf_dict["Sparsity of e2"] = []
    tf_dict["Sparsity of e3"] = []
    tf_dict["basis"] = []
    tf_dict["state_case_index"] = []
    tf_dict["base_case_index"] = []

    state_case_index = 0
    while os.path.exists(dp_root.format(state_case_index, 0)):
        base_case_index = 0
        while os.path.exists(dp_root.format(state_case_index, base_case_index)):
            directory = dp_root.format(state_case_index, base_case_index)
            tf_dict["state_case_index"].append(state_case_index)
            tf_dict["base_case_index"].append(base_case_index)

            fidelity_track_array = []
            trial_index = 0
            while os.path.isfile(directory + "/fidelity_tracks_%03d" % trial_index):

                states, tracked_fidelity = pr.parse_fidelity_track(directory + "/fidelity_tracks_%03d" % trial_index)
                fidelity_track_array.append(tracked_fidelity)
                trial_index += 1

            tf_dict["basis"].append(states)
            sc = pr.get_success_criteria(fidelity_track_array, reference_stage=ref_stage, g=g)
            for key, element in sc.items():
                tf_dict[key].append(element)
            tf_dict["Sparsity of the basis"].append(pr.get_basis_sparseness(states))
            for i in range(4):
                tf_dict["Sparsity of e"+str(i)].append(pr.get_basis_sparseness([states[i, :]]))

            print("Case %03d - %03d" % (state_case_index, base_case_index))
            print("Sparsity of the basis: %.3f" % pr.get_basis_sparseness(states))
            print()
            for key, value in sc.items():
                print(key, value, sep=":\t")
            print("\n"+"-"*32+"\n")

            base_case_index += 1
        state_case_index += 1

    df = pd.DataFrame.from_dict(tf_dict)
    print(df)
    if ref_stage == 1:
        df.to_csv(bigdir + '/' + bigdir + "_dataframe")
        print("Saved in path:\n" + bigdir+'/'+bigdir+'_dataframe')
    else:
        df.to_csv(bigdir + '/' + bigdir + "_dataframe"+"_%d" % ref_stage)
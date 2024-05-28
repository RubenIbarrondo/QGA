This is the archived version of the project. 

It is kept as a branch to facilitate access to the original code used in 

- [https://doi.org/10.1109/TEVC.2023.3296780]()
- [https://doi.org/10.1109/SSCI51031.2022.10022159]()


-----

# Quantum Genetic Algorithm

This are programs for the simulation of the quantum genetic algorithm (QGA).

- [quantum_mats]() - It contains self created classes to work with the quantum operators needed for the QGA.
Often, these operators are represented by matrix with a high dimensionality, but can be sometimes simulated efficiently employing their properties.

- [QGA_BCQO_sim]() and [QGA_UQCM_sim]() - They contain the functions that simulate the QGA process with different replication subroutines.

- [QGA_BCQO_quantum_channels]() and [QGA_UQCM_quantum_channels]() - They contain the functions required to compute the Kraus matrices for the quantum channels representing the QGA and the functions to compute the fixed points of the channels.

- [QGA_BCQO_fixed_point_test]() - It performs a comparison between the results of the simulations and the fixed point analysis. We could only implement it for BCQO-based QGA.
 
- [QGA_performance_report.py]() - It contains a set of functions to get performance ratios or to show graphically some characteristics of the outputs of the QGA.

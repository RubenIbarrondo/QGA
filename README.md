# Quantum Genetic Algorithm

This are programs for the simulation of the quantum genetic algorithm (QGA). There are three script files:

- [quantum_mats]() - It contains self created classes to work with the quantum operators needed for the QGA.
This operators are represented by matrix with a high dimensionality, but can be simulated effectively employing their properties.

- [QGA_vs_CGA]() -  It contains the functions `QuantumGeneticAlgorithm` and `ClassicGeneticAlgorithm`,
 which implement the QGA and a classical analog (CGA). The quantum version employs the classes in `quantum_mats`.
 Both generate output files.
 
- [QGA_performance_report.py]() - It contains a set of functions to get performance ratios or to show graphically some characteristics of the outputs of the QGA.

The directory [Notebooks]() contains some notebooks used during the development of the programs.
The other directories contain the outputs of QGA and CGA that have been tested with different initial populations and fitness functions.
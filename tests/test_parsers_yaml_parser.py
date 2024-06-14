import unittest

import sys, os


from src.subroutines import cloning, sorting, mutation, mixing
from src.generators import problem_generators, state_generators
from src import feature_trackers

from src.parsers import yaml_parser


class TestParsers_YamlParser(unittest.TestCase):

    def setUp(self) -> None:
        self.maxDiff = None

        # set source path
        self.file_path = os.path.join(sys.path[0], 'tests', 'fixtures', 'input_testing.yaml')

        # create ref dicts
        # Values to pystrings
        self.vpystrings = []
        self.vpystrings.append({'input-id': 'test.uqcm.20.05.2024', 'output-directory-path': 'outputs', 'track-features': ['IndividualEigenstateFidelity'], 'population-size': 4, 'chromosome-size': 2, 'problem-generating-procedure': 'HaarRandomHamiltonian', 'problem-generating-procedure-attributes': {'energies': 'non-degenerate', 'problem-instance-number': 5, 'seed': None}, 'initial-state-generating-procedure': 'HaarRandomStates', 'initial-state-generating-procedure-attributes': {'initial-population-number': 10, 'seed': None}, 'qga-attributes': {'cloning': 'UQCM', 'mixing': 'MixingOff', 'sorting': 'FullSort', 'sorting-attributes': {'problem-deffinition': 'hamiltonian'}, 'mutation': 'RandomPauli', 'mutation-attributes': {'mutation-probability': 0.1}, 'generations': 10}})
        self.vpystrings.append({'input-id': 'test.uqcm.20.05.2024.pseudo-reproducible', 'output-directory-path': 'outputs', 'track-features': ['IndividualEigenstateFidelity'], 'population-size': 4, 'chromosome-size': 2, 'problem-generating-procedure': 'HaarRandomHamiltonian', 'problem-generating-procedure-attributes': {'dimension': 4, 'energies': 'non-degenerate', 'problem-instance-number': 5, 'seed': 123}, 'initial-state-generating-procedure': 'HaarRandomStates', 'initial-state-generating-procedure-attributes': {'initial-population-number': 10, 'seed': 123}, 'qga-attributes': {'cloning': 'UQCM', 'mixing': 'MixingOff', 'sorting': 'FullSort', 'sorting-attributes': {'problem-deffinition': 'hamiltonian'}, 'mutation': 'RandomPauli', 'mutation-attributes': {'mutation-probability': 0.1}, 'generations': 10}})
        self.vpystrings.append({'input-id': 'test.uqcm.20.05.2024.dir-reproducible', 'output-directory-path': 'outputs', 'track-features': ['IndividualEigenstateFidelity'], 'population-size': 4, 'chromosome-size': 2, 'problem-generating-procedure': 'HamiltonianSampleDirectory', 'problem-generating-procedure-attributes': {'dirpath': 'hamiltonians'}, 'initial-state-generating-procedure': 'InitStateSampleDirectory', 'initial-state-generating-procedure-attributes': {'dirpath': 'initial_states'}, 'qga-attributes': {'cloning': 'UQCM', 'mixing': 'MixingOff', 'sorting': 'FullSort', 'sorting-attributes': {'problem-deffinition': 'hamiltonian'}, 'mutation': 'RandomPauli', 'mutation-attributes': {'mutation-probability': 0.1}, 'generations': 10}})
        
        # Attributes to pystrings
        self.apystrings = []
        self.apystrings.append({'input_id': 'test.uqcm.20.05.2024', 'output_directory_path': 'outputs',
                                'track_features': ['IndividualEigenstateFidelity'],
                                'track_features_attributes': {'chromosome_size': 2, 'population_size': 4},
                                'population_size': 4, 'chromosome_size': 2,
                                'problem_generating_procedure': 'HaarRandomHamiltonian', 'problem_generating_procedure_attributes': {'energies': 'non-degenerate', 'problem_instance_number': 5, 'seed': None, 'population_size': 4, 'chromosome_size': 2},
                                'initial_state_generating_procedure': 'HaarRandomStates', 'initial_state_generating_procedure_attributes': {'initial_population_number': 10, 'seed': None, 'population_size': 4, 'chromosome_size': 2},
                                'qga_attributes': {'cloning': 'UQCM', 'cloning_attributes': {'population_size': 4, 'chromosome_size': 2},
                                                    'mixing': 'MixingOff', 'mixing_attributes': {'population_size': 4, 'chromosome_size': 2},
                                                    'sorting': 'FullSort', 'sorting_attributes': {'problem_deffinition': 'hamiltonian', 'population_size': 4, 'chromosome_size': 2},
                                                    'mutation': 'RandomPauli', 'mutation_attributes': {'mutation_probability': 0.1, 'population_size': 4, 'chromosome_size': 2},
                                                    'generations': 10, 'population_size': 4, 'chromosome_size': 2}})
        self.apystrings.append({'input_id': 'test.uqcm.20.05.2024.pseudo-reproducible', 'output_directory_path': 'outputs',
                                'track_features': ['IndividualEigenstateFidelity'],
                                'track_features_attributes': {'chromosome_size': 2, 'population_size': 4},
                                'population_size': 4, 'chromosome_size': 2,
                                'problem_generating_procedure': 'HaarRandomHamiltonian', 'problem_generating_procedure_attributes': {'dimension': 4, 'energies': 'non-degenerate', 'problem_instance_number': 5, 'seed': 123, 'population_size': 4, 'chromosome_size': 2},
                                'initial_state_generating_procedure': 'HaarRandomStates', 'initial_state_generating_procedure_attributes': {'initial_population_number': 10, 'seed': 123, 'population_size': 4, 'chromosome_size': 2},
                                'qga_attributes': {'cloning': 'UQCM', 'cloning_attributes': {'population_size': 4, 'chromosome_size': 2},
                                                    'mixing': 'MixingOff', 'mixing_attributes': {'population_size': 4, 'chromosome_size': 2},
                                                    'sorting': 'FullSort', 'sorting_attributes': {'problem_deffinition': 'hamiltonian', 'population_size': 4, 'chromosome_size': 2},
                                                    'mutation': 'RandomPauli', 'mutation_attributes': {'mutation_probability': 0.1, 'population_size': 4, 'chromosome_size': 2},
                                                    'generations': 10, 'population_size': 4, 'chromosome_size': 2}})
        self.apystrings.append({'input_id': 'test.uqcm.20.05.2024.dir-reproducible', 'output_directory_path': 'outputs',
                                'track_features': ['IndividualEigenstateFidelity'],
                                'track_features_attributes': {'chromosome_size': 2, 'population_size': 4},
                                'population_size': 4, 'chromosome_size': 2,
                                'problem_generating_procedure': 'HamiltonianSampleDirectory', 'problem_generating_procedure_attributes': {'dirpath': 'hamiltonians', 'population_size': 4, 'chromosome_size': 2},
                                'initial_state_generating_procedure': 'InitStateSampleDirectory', 'initial_state_generating_procedure_attributes': {'dirpath': 'initial_states', 'population_size': 4, 'chromosome_size': 2},
                                'qga_attributes': {'cloning': 'UQCM', 'cloning_attributes': {'population_size': 4, 'chromosome_size': 2},
                                                    'mixing': 'MixingOff', 'mixing_attributes': {'population_size': 4, 'chromosome_size': 2},
                                                    'sorting': 'FullSort', 'sorting_attributes': {'problem_deffinition': 'hamiltonian', 'population_size': 4, 'chromosome_size': 2},
                                                    'mutation': 'RandomPauli', 'mutation_attributes': {'mutation_probability': 0.1, 'population_size': 4, 'chromosome_size': 2},
                                                    'generations': 10, 'population_size': 4, 'chromosome_size': 2}})
        
        # Strings to Python objects
        self.pyobjects = []
        self.pyobjects.append({'input_id': 'test.uqcm.20.05.2024', 'output_directory_path': 'outputs',
                               'track_features': [feature_trackers.IndividualEigenstateFidelity(population_size=4, chromosome_size = 2)],
                                'track_features_attributes': {'chromosome_size': 2, 'population_size': 4},
                               'population_size': 4, 'chromosome_size': 2,
                           'problem_generating_procedure': problem_generators.HaarRandomHamiltonian(energies = 'non-degenerate', problem_instance_number=5, seed=None, population_size = 4, chromosome_size = 2), 'problem_generating_procedure_attributes': {'energies': 'non-degenerate', 'problem_instance_number': 5, 'seed': None, 'population_size': 4, 'chromosome_size': 2},
                           'initial_state_generating_procedure':state_generators.HaarRandomStates(initial_population_number = 10, seed = None, population_size = 4, chromosome_size = 2), 'initial_state_generating_procedure_attributes': {'initial_population_number': 10, 'seed': None, 'population_size': 4, 'chromosome_size': 2},
                           'qga_attributes': {'cloning': cloning.UQCM(population_size=4, chromosome_size = 2), 'cloning_attributes': {'population_size': 4, 'chromosome_size': 2},
                                              'mixing': mixing.MixingOff(population_size=4, chromosome_size = 2), 'mixing_attributes': {'population_size': 4, 'chromosome_size': 2},
                                              'sorting': sorting.FullSort(problem_deffinition = 'hamiltonian', population_size = 4, chromosome_size = 2), 'sorting_attributes': {'problem_deffinition': 'hamiltonian', 'population_size': 4, 'chromosome_size': 2},
                                              'mutation': mutation.RandomPauli(mutation_probability = 0.1, population_size = 4, chromosome_size = 2), 'mutation_attributes': {'mutation_probability': 0.1, 'population_size': 4, 'chromosome_size': 2},
                                              'generations': 10, 'population_size': 4, 'chromosome_size': 2}})
        self.pyobjects.append({'input_id': 'test.uqcm.20.05.2024.pseudo-reproducible', 'output_directory_path': 'outputs',
                               'track_features': [feature_trackers.IndividualEigenstateFidelity(population_size=4, chromosome_size = 2)],
                                'track_features_attributes': {'chromosome_size': 2, 'population_size': 4},
                               'population_size': 4, 'chromosome_size': 2,
                           'problem_generating_procedure': problem_generators.HaarRandomHamiltonian(energies = 'non-degenerate', problem_instance_number=5, seed=123, population_size = 4, chromosome_size = 2), 'problem_generating_procedure_attributes': {'dimension': 4, 'energies': 'non-degenerate', 'problem_instance_number': 5, 'seed': 123, 'population_size': 4, 'chromosome_size': 2},
                           'initial_state_generating_procedure': state_generators.HaarRandomStates(initial_population_number = 10, seed = 123, population_size = 4, chromosome_size = 2), 'initial_state_generating_procedure_attributes': {'initial_population_number': 10, 'seed': 123, 'population_size': 4, 'chromosome_size': 2},
                           'qga_attributes': {'cloning': cloning.UQCM(population_size=4, chromosome_size = 2), 'cloning_attributes': {'population_size': 4, 'chromosome_size': 2},
                                              'mixing': mixing.MixingOff(population_size=4, chromosome_size = 2), 'mixing_attributes': {'population_size': 4, 'chromosome_size': 2},
                                              'sorting': sorting.FullSort(problem_deffinition = 'hamiltonian', population_size = 4, chromosome_size = 2), 'sorting_attributes': {'problem_deffinition': 'hamiltonian', 'population_size': 4, 'chromosome_size': 2},
                                              'mutation': mutation.RandomPauli(mutation_probability = 0.1, population_size = 4, chromosome_size = 2), 'mutation_attributes': {'mutation_probability': 0.1, 'population_size': 4, 'chromosome_size': 2},
                                              'generations': 10, 'population_size': 4, 'chromosome_size': 2}})
        self.pyobjects.append({'input_id': 'test.uqcm.20.05.2024.dir-reproducible', 'output_directory_path': 'outputs',
                               'track_features': [feature_trackers.IndividualEigenstateFidelity(population_size=4, chromosome_size = 2)],
                                'track_features_attributes': {'chromosome_size': 2, 'population_size': 4},
                                'population_size': 4, 'chromosome_size': 2,
                           'problem_generating_procedure': problem_generators.HamiltonianSampleDirectory(dirpath = 'hamiltonians', population_size = 4, chromosome_size = 2), 'problem_generating_procedure_attributes': {'dirpath': 'hamiltonians', 'population_size': 4, 'chromosome_size': 2},
                           'initial_state_generating_procedure': state_generators.InitStateSampleDirectory(dirpath = 'initial_states', population_size = 4, chromosome_size = 2), 'initial_state_generating_procedure_attributes': {'dirpath': 'initial_states', 'population_size': 4, 'chromosome_size': 2},
                           'qga_attributes': {'cloning': cloning.UQCM(population_size=4, chromosome_size = 2), 'cloning_attributes': {'population_size': 4, 'chromosome_size': 2},
                                              'mixing': mixing.MixingOff(population_size=4, chromosome_size = 2), 'mixing_attributes': {'population_size': 4, 'chromosome_size': 2},
                                              'sorting': sorting.FullSort(problem_deffinition = 'hamiltonian', population_size = 4, chromosome_size = 2), 'sorting_attributes': {'problem_deffinition': 'hamiltonian', 'population_size': 4, 'chromosome_size': 2},
                                              'mutation': mutation.RandomPauli(mutation_probability = 0.1, population_size = 4, chromosome_size = 2), 'mutation_attributes': {'mutation_probability': 0.1, 'population_size': 4, 'chromosome_size': 2},
                                              'generations': 10, 'population_size': 4, 'chromosome_size': 2}})


    def test_parse_values_to_pystrings(self):
        yaml_data_list = yaml_parser.load_yaml_data(self.file_path)
        
        for yaml_data, input_data_ref in zip(yaml_data_list, self.vpystrings):
            input_data = yaml_parser.parse_values_to_pystrings(yaml_data)
            self.assertEqual(input_data, input_data_ref)
        
    def test_parse_attribute_dicts(self):
        for input_data_vpystrings, input_data_ref in zip(self.vpystrings, self.apystrings):
            input_data = yaml_parser.parse_attribute_dicts(input_data_vpystrings)
            self.assertEqual(input_data, input_data_ref)

    def test_parse_to_python_objects(self):
        for input_data_apystrings, input_data_ref in zip(self.apystrings, self.pyobjects):
            input_data = yaml_parser.parse_to_python_objects(input_data_apystrings)
            self.assertEqual(input_data, input_data_ref)

    def test_parse_yaml(self):
        input_data_list = yaml_parser.parse_yaml(self.file_path)
        for input_data, input_data_ref in zip(input_data_list, self.pyobjects):
            self.assertEqual(input_data, input_data_ref)

    def test_parse_to_python_objects_problem_generator(self):
        for input_data_apystrings, input_data_ref in zip(self.apystrings, self.pyobjects):
            input_data = yaml_parser.parse_to_python_objects(input_data_apystrings)
            self.assertEqual(input_data["problem_generating_procedure"],
                              input_data_ref["problem_generating_procedure"],
                              msg=f"{input_data['problem_generating_procedure'].__dict__} != {input_data_ref['problem_generating_procedure'].__dict__}")
            
    def test_parse_to_python_objects_state_generator(self):
        for input_data_apystrings, input_data_ref in zip(self.apystrings, self.pyobjects):
            input_data = yaml_parser.parse_to_python_objects(input_data_apystrings)
            self.assertEqual(input_data["initial_state_generating_procedure"],
                              input_data_ref["initial_state_generating_procedure"],
                              msg=f"{input_data['initial_state_generating_procedure'].__dict__} != {input_data_ref['initial_state_generating_procedure'].__dict__}")
            
    def test_parse_to_python_objects_qga_attributes(self):
        for input_data_apystrings, input_data_ref in zip(self.apystrings, self.pyobjects):
            input_data = yaml_parser.parse_to_python_objects(input_data_apystrings)
            self.assertTrue(all([value == input_data_ref["qga_attributes"][key] for key, value in input_data["qga_attributes"].items()]))
                
    def test_parse_to_python_objects_qga_attributes_sorting(self):
        for input_data_apystrings, input_data_ref in zip(self.apystrings, self.pyobjects):
            input_data = yaml_parser.parse_to_python_objects(input_data_apystrings)
            self.assertEqual(input_data["qga_attributes"]['sorting'],
                              input_data_ref["qga_attributes"]['sorting'],
                              msg=f"{input_data['qga_attributes']['sorting'].__dict__} != {input_data_ref['qga_attributes']['sorting'].__dict__}")
    
    def test_parse_to_python_objects_qga_attributes_mutation(self):
        for input_data_apystrings, input_data_ref in zip(self.apystrings, self.pyobjects):
            input_data = yaml_parser.parse_to_python_objects(input_data_apystrings)
            self.assertEqual(input_data["qga_attributes"]['mutation'],
                              input_data_ref["qga_attributes"]['mutation'],
                              msg=f"{input_data['qga_attributes']['mutation'].__dict__} != {input_data_ref['qga_attributes']['mutation'].__dict__}")
            
    def test_parse_to_python_objects_qga_attributes_cloning(self):
        for input_data_apystrings, input_data_ref in zip(self.apystrings, self.pyobjects):
            input_data = yaml_parser.parse_to_python_objects(input_data_apystrings)
            self.assertEqual(input_data["qga_attributes"]['cloning'],
                              input_data_ref["qga_attributes"]['cloning'],
                              msg=f"{input_data['qga_attributes']['cloning'].__dict__} != {input_data_ref['qga_attributes']['cloning'].__dict__}")
            
    def test_parse_to_python_objects_qga_attributes_mixing(self):
        for input_data_apystrings, input_data_ref in zip(self.apystrings, self.pyobjects):
            input_data = yaml_parser.parse_to_python_objects(input_data_apystrings)
            self.assertEqual(input_data["qga_attributes"]['mixing'],
                              input_data_ref["qga_attributes"]['mixing'],
                              msg=f"{input_data['qga_attributes']['mixing'].__dict__} != {input_data_ref['qga_attributes']['mixing'].__dict__}")
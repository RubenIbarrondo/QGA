import sys
from quantum_genetic_algorithm import quantum_genetic_algorithm
from parsers.yalm_parser import parse_yalm
from parsers.output_saver import save_outputs


if __name__ == "__main__":

    # Read the input YAML
    input_instances = parse_yalm(sys.argv[1])

    # For page in the YAML run the corresponding simulation
    for input_instance in input_instances:

        # Get problem instance generator
        problem_gen = input_instance["problem-generating-procedure"](**input_instance["problem-generating-procedure-attributes"])
        
        for id_problem, problem in enumerate(problem_gen):
            # The sorting procedure should be updated here to include the info about the Hamiltonian
            input_instance['qga-attributes']['sorting'].set_problem(problem)

            # Get state generator
            state_gen = input_instance["initial-state-generating-procedure"](**input_instance["initial-state-generating-procedure-attributes"])
            
            for id_state, state in enumerate(state_gen):
                
                outputs = quantum_genetic_algorithm(initial_state = state,
                                                    track_features = input_instance["track-features"],
                                                    **input_instance["qga-attributes"])
                
                save_outputs(outputs, 
                             output_directory_path = input_instance["output-directory-path"],
                             input_id = input_instance["input-id"],
                             problem_instance_id = id_problem,
                             state_instance_id = id_state)
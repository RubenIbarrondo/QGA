import sys
from qga_toolbox.quantum_genetic_algorithm import quantum_genetic_algorithm
from parsers.yaml_parser import parse_yaml
from parsers.output_saver import save_outputs


def run_qga_simulation(input_yaml_path: str):
    # Read the input YAML
    input_instances = parse_yaml(input_yaml_path)

    # For page in the YAML run the corresponding simulation
    for input_instance in input_instances:
        
        # Extract generators
        prolem_generator = input_instance["problem_generating_procedure"]
        state_generator = input_instance["initial_state_generating_procedure"]
        
        for id_problem, problem in enumerate(prolem_generator.generate()):
            # The sorting procedure should be updated here to include the Hamiltonian
            input_instance['qga_attributes']['sorting'].set_problem(problem)
            
            for id_state, state in enumerate(state_generator.generate()):
                
                outputs = quantum_genetic_algorithm(initial_state = state,
                                                    track_features = input_instance["track_features"],
                                                    **input_instance["qga_attributes"])
                
                save_outputs(outputs, 
                             output_directory_path = input_instance["output_directory_path"],
                             input_id = input_instance["input_id"],
                             problem_instance_id = id_problem,
                             state_instance_id = id_state)

if __name__ == "__main__":
    run_qga_simulation(sys.argv[1])

    

    
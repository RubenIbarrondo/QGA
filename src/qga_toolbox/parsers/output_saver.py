import os
import numpy as np


def save_outputs(outputs, output_directory_path: str, input_id: str, problem_instance_id: int, state_instance_id: int):
    
    # check collisions in outputs...
    final_state, tracked_features = outputs
    tracked_features['state'] = final_state

    if not os.path.exists(output_directory_path):
        os.mkdir(output_directory_path)
    if not os.path.exists(os.path.join(output_directory_path, input_id)):
        os.mkdir(os.path.join(output_directory_path, input_id))

    try:        
        for feature_name, feature_arr in tracked_features.items():
            file_name = f"problem_{problem_instance_id}_state_{state_instance_id}_{feature_name}"
            
            np.save(os.path.join(output_directory_path, input_id, file_name),
                    np.array(feature_arr))
            
    except FileNotFoundError as fnfe:
        raise fnfe
        


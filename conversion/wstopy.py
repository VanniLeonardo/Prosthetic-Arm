import scipy.io
import numpy as np
import os
import json

# Load the MAT file
file_path = "/Users/giovanninocerino/Desktop/Associations/BAINSA/EEG/grasp-and-lift-eeg-detection/P5/WS_P5_S1.mat" # Replace with your file path
mat_data = scipy.io.loadmat(file_path)

# Extract the 'win' data from 'ws'
ws_content = mat_data['ws'][0, 0]
win_data = ws_content['win'][0]  # Extract the actual list of 28 experiments

# Define the structured dictionary
structured_data = {
    "experiments": []
}

# Iterate through each of the 28 experiments and structure it properly
for experiment in win_data:
    experiment_data = {
        "eeg": experiment[0].tolist() if isinstance(experiment[0], np.ndarray) else None,
        "kinematics": experiment[1].tolist() if isinstance(experiment[1], np.ndarray) else None,
        "emg": experiment[2].tolist() if isinstance(experiment[2], np.ndarray) else None,
        "timestamps": {
            "eeg_t": experiment[3].tolist() if isinstance(experiment[3], np.ndarray) else None,
            "emg_t": experiment[4].tolist() if isinstance(experiment[4], np.ndarray) else None,
            "trial_start": experiment[5].tolist() if isinstance(experiment[5], np.ndarray) else None,
            "trial_end": experiment[8].tolist() if isinstance(experiment[8], np.ndarray) else None,
            "LEDon": experiment[6].tolist() if isinstance(experiment[6], np.ndarray) else None,
            "LEDoff": experiment[7].tolist() if isinstance(experiment[7], np.ndarray) else None,
        },
        "experimental_conditions": {
            "weight": experiment[9].tolist() if isinstance(experiment[9], np.ndarray) else None,
            "weight_id": experiment[10].tolist() if isinstance(experiment[10], np.ndarray) else None,
            "surface": experiment[11].tolist() if isinstance(experiment[11], np.ndarray) else None,
            "surface_id": experiment[12].tolist() if isinstance(experiment[12], np.ndarray) else None,
            "previous_weight": experiment[13].tolist() if isinstance(experiment[13], np.ndarray) else None,
            "previous_weight_id": experiment[14].tolist() if isinstance(experiment[14], np.ndarray) else None,
            "previous_surface": experiment[15].tolist() if isinstance(experiment[15], np.ndarray) else None,
            "previous_surface_id": experiment[16].tolist() if isinstance(experiment[16], np.ndarray) else None,
        },
    }
    structured_data["experiments"].append(experiment_data)

#Debugging print
print(structured_data["experiments"][4]["experimental_conditions"]["weight_id"])

# Generate JSON filename based on input file
json_filename = os.path.splitext(file_path)[0] + ".json"

# Write JSON file
with open(json_filename, "w") as json_file:
    json.dump(structured_data, json_file, indent=4)

print(f"JSON saved: {json_filename}")

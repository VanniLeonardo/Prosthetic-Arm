import scipy.io
import numpy as np
import os
import json

# Load the .mat file
file_path = "/Users/giovanninocerino/Desktop/Associations/BAINSA/EEG/grasp-and-lift-eeg-detection/P5/P5_AllLifts.mat"  # Replace with your file path
mat_data = scipy.io.loadmat(file_path)

# Extract the 'P' variable
P_content = mat_data['P'][0, 0]

# Extract column names from the metadata array
column_names = [str(name_array[0]) for name_array in P_content[1].flatten()]

# Extract numerical data
numerical_data = P_content[0]

# Convert data to a structured dictionary
structured_data = {
    "columns": column_names,
    "data": numerical_data.tolist()  # 244 points of data
}

# Generate JSON filename based on input file
json_filename = os.path.splitext(file_path)[0] + ".json"

# Write JSON file
with open(json_filename, "w") as json_file:
    json.dump(structured_data, json_file, indent=4)

print(f"JSON saved: {json_filename}")

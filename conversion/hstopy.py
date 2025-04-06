import scipy.io
import numpy as np
import os
import json



file_path = "/Users/giovanninocerino/Desktop/Associations/EEG Research/math_file/P1/HS_P1_ST.mat" # Replace with your file path

# Check if the file exists before proceeding
if not os.path.exists(file_path):
    raise FileNotFoundError(f"Error: The file '{file_path}' was not found. Please check the path and try again.")

mat_data = scipy.io.loadmat(file_path)

# Extract the 'hs' variable
hs_data = mat_data.get('hs')

# Function to extract signal data, names, and sampling rates
def extract_signal_data(section, key):
    try:
        struct = section[key][0, 0]
        data = np.array(struct['sig'])  # Signal data

        # Extract names correctly
        for namelist in (struct["names"]):
            #print("ciao", name)
            if isinstance(namelist, np.ndarray) and namelist.dtype == 'O':
                names = [name[0] for name in namelist if isinstance(name, np.ndarray) and len(name) > 0]
            else:
                names = [namelist.item()]

        # Extract sampling rate
        sampling_rate = struct['samplingrate'][0, 0].item()  # Extract sampling rate
        #print(f"{key} Names Extracted: {names}")  # Debugging print
        return data.tolist(), names, sampling_rate  # Convert NumPy array to list
    except Exception as e:
        print(f"Error extracting {key}: {e}")
        return [], [], None  # Return empty lists instead of NumPy arrays

# Ensure 'hs_data' exists before accessing it
if hs_data is None:
    raise ValueError("Error: 'hs' variable not found in the .mat file. Please check the file contents.")

# Extracting each data type properly
emg_data, emg_names, emg_sampling_rate = extract_signal_data(hs_data[0, 0], 'emg')
eeg_data, eeg_names, eeg_sampling_rate = extract_signal_data(hs_data[0, 0], 'eeg')
kin_data, kin_names, kin_sampling_rate = extract_signal_data(hs_data[0, 0], 'kin')
env_data, env_names, env_sampling_rate = extract_signal_data(hs_data[0, 0], 'env')
misc_data, misc_names, misc_sampling_rate = extract_signal_data(hs_data[0, 0], 'misc')

# Ensure EMG sampling rate is set correctly if missing
if emg_sampling_rate is None or emg_sampling_rate == 0:
    emg_sampling_rate = 4000  # Assuming based on initial inspection

# Prepare structured dictionary
structured_data = {
    "EMG": {"data": emg_data, "names": emg_names, "sampling_rate": emg_sampling_rate},
    "EEG": {"data": eeg_data, "names": eeg_names, "sampling_rate": eeg_sampling_rate},
    "KIN": {"data": kin_data, "names": kin_names, "sampling_rate": kin_sampling_rate},
    "ENV": {"data": env_data, "names": env_names, "sampling_rate": env_sampling_rate},
    "MISC": {"data": misc_data, "names": misc_names, "sampling_rate": misc_sampling_rate},
}

print(structured_data["ENV"]["names"])



# Generate JSON filename based on input file
json_filename = os.path.splitext(file_path)[0] + ".json"

#Write JSON file
with open(json_filename, "w") as json_file:
    json.dump(structured_data, json_file, indent=4)

print(f"JSON saved: {json_filename}")

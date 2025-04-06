import os
import numpy as np
from scipy.io import loadmat

# === Directory containing all participant folders ===
base_path = "/Users/giovanninocerino/Desktop/Associations/EEG Research/math_file"

# === Iterate through participants P1 to P12 ===
for p in range(1, 13):
    participant_folder = os.path.join(base_path, f"P{p}")
    mat_file = os.path.join(participant_folder, f"HS_P{p}_ST.mat")

    if not os.path.exists(mat_file):
        print(f"[!] File not found: {mat_file}")
        continue

    try:
        # Load .mat file
        mat_data = loadmat(mat_file)

        # Access 'hs' structure
        if 'hs' not in mat_data:
            print(f"[!] 'hs' key missing in {mat_file}")
            continue

        hs_struct = mat_data['hs']

        # Navigate to 'eeg' field
        eeg_struct = hs_struct['eeg'][0, 0]

        # Extract EEG signal from 'sig'
        eeg_signal = eeg_struct['sig'][0, 0]

        # Save to .npy file
        out_path = os.path.join(participant_folder, f"HS_P{p}_ST_eeg.npy")
        np.save(out_path, eeg_signal)

        print(f"[✓] Saved EEG for P{p} to {out_path} — shape: {eeg_signal.shape}")

    except Exception as e:
        print(f"[!] Failed to process P{p}: {e}")

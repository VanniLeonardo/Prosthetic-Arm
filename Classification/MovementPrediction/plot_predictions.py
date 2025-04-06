
import os
import numpy as np
import matplotlib.pyplot as plt

# === SETTINGS ===
participant = 3
base_path = "/Users/giovanninocerino/Desktop/Associations/EEG Research/math_file"
window_size = 500
step = 250
channel_to_plot = 2  # you can change this to view different channels

# === Load EEG and Predictions ===
eeg_path = os.path.join(base_path, f"P{participant}", f"HS_P{participant}_S_eeg.npy")
pred_path = os.path.join(base_path, f"P{participant}", f"P{participant}_predictions.npy")

eeg = np.load(eeg_path)  # shape: (T, C)
preds = np.load(pred_path)  # shape: (N,)

# Transpose to (C, T)
eeg = eeg.T

# Get the EEG for the selected channel
eeg_ch = eeg[channel_to_plot]

# Reconstruct prediction window ranges (start/end indices)
movement_mask = np.zeros_like(eeg_ch)
for i, pred in enumerate(preds):
    if pred == 1:
        start = i * step
        end = start + window_size
        movement_mask[start:end] = 1

# === Plot ===
plt.figure(figsize=(15, 5))
plt.plot(eeg_ch, label=f"EEG Channel {channel_to_plot}")
plt.fill_between(np.arange(len(eeg_ch)), eeg_ch.min(), eeg_ch.max(), where=movement_mask > 0, 
                 color='red', alpha=0.2, label='Predicted Movement')
plt.title(f"P{participant} – EEG Signal (Channel {channel_to_plot})")
plt.xlabel("Time (samples)")
plt.ylabel("Amplitude")
plt.legend()
plt.tight_layout()
plt.show()

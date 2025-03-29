# The strategy is to use correspondance between WS and HS data. Label 1 all the correspective point in HS that
# correspond to the WS data, zero everywhere else. Use this label to train the model to recognize movement.
# Data has to be windowed to enter EEGNet.
# For more questions text me.

import orjson
import json
import numpy as np
from EEGModels import EEGNet

s_r = 500 # Sampling Rate
def extract_window_movements(p,s):
        filepath = f"/Users/giovanninocerino/Desktop/Associations/EEG Research/math_file/P{p}/WS_P{p}_S{s}.json" # Replace with your file path
        with open(filepath, "rb") as f:
            data = json.load(f)
        time_spans = []
        trial_counter = 0
        for trial in data["experiments"]:
            start = int(trial["timestamps"]["trial_start"][0][0] * s_r) - 999 #999 was found by looking at the dataset
            end = int(trial["timestamps"]["trial_end"][0][0] * s_r) - 999
            time_spans.append([start,end])
        spans = np.array(time_spans)
        return spans

def extract_hs(p,s):
    filepath = f"/Users/giovanninocerino/Desktop/Associations/EEG Research/math_file/P{p}/HS_P{p}_S{s}.json" # Replace with your file path
    with open(filepath, "rb") as f:
        data = json.load(f)
    eeg_signal = data["EEG"]["data"]
    # Convert to numpy array
    eeg_signal = np.array(eeg_signal)
    return eeg_signal

def assign_labels(p, s):
    eeg = extract_hs(p, s)
    labels = np.zeros(len(eeg), dtype=int)
    time_spans = extract_window_movements(p, s)
    for start, end in time_spans:
        labels[start:end] = 1  

    return labels


def create_epochs(eeg, labels, window_size=500, step=250, threshold=0.5):
    """
    eeg:  (T, C) array (time x channel)
    labels: (T,) array with 0 or 1 for each time sample
    window_size: number of samples per window
    step: how many samples to move each iteration
    threshold: fraction of labeled-1 samples needed to consider the window 'movement'
    """
    T, channels = eeg.shape
    X_list = []
    y_list = []

    for start_idx in range(0, T - window_size, step):
        end_idx = start_idx + window_size
        # slice the time dimension
        window_data = eeg[start_idx:end_idx, :]  # shape => (window_size, C)
        window_labels = labels[start_idx:end_idx] # shape => (window_size,)

        #average of the labels in the window
        # if the average is greater than threshold, assign label 1
        frac_movement = np.mean(window_labels)
        window_label = 1 if frac_movement >= threshold else 0

        # now if you want shape (C, window_size) for EEGNet, transpose it:
        window_data = window_data.T  # => (C, window_size)

        X_list.append(window_data)
        y_list.append(window_label)

    X = np.array(X_list)  # => (num_windows, C, window_size)
    y = np.array(y_list)
    return X, y

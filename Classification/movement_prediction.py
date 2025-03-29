# The strategy is to use correspondance between WS and HS data. Label 1 all the correspective point in HS that
# correspond to the WS data, zero everywhere else. Use this label to train the model to recognize movement.
# Data has to be windowed to enter EEGNet.
# For more questions text me.

import orjson
import json
import numpy as np
from EEGModels import EEGNet
import tensorflow as ts
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical

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

def gather_all_data(participants, sessions, window_size=500, step=250, threshold=0.5):
    """
    Returns X_all, y_all from *all* participants and sessions.
    Each row in X_all is one epoch (C, window_size),
    y_all is the corresponding label (0 or 1).
    """
    X_list = []
    y_list = []
    
    for p in participants:
        for s in sessions:
            print(f"Gathering data for P{p}, S{s}")
            # 1) Extract continuous EEG
            eeg = extract_hs(p, s)   # shape => (T, C) or (C, T) depending on your data
            # 2) Get sample-wise labels
            labels = assign_labels(p, s)  # shape => (T,)
            # 3) Window into epochs
            X, y = create_epochs(eeg, labels,
                                 window_size=window_size,
                                 step=step,
                                 threshold=threshold)
            # 4) Collect
            X_list.append(X)
            y_list.append(y)
    
    # Concatenate
    # X_list is a list of arrays, each shape => (num_windows_ps, C, window_size)
    X_all = np.concatenate(X_list, axis=0)  # e.g. (N_total, C, window_size)
    y_all = np.concatenate(y_list, axis=0)  # (N_total,)
    
    print(f"All data shape: {X_all.shape}, labels shape: {y_all.shape}")
    return X_all, y_all

X_all, y_all = gather_all_data(
    participants = [1,2,3,4, 5, 6, 7, 8, 9, 10, 11, 12],  
    sessions     = [1,2, 3, 4, 5, 6, 7, 8, 9],   
    window_size  = 500,
    step         = 250,
    threshold    = 0.5
)

X_train, X_val, y_train, y_val = train_test_split(X_all, y_all, test_size=0.2, random_state=42)


X_train = np.expand_dims(X_train, axis=-1)  # => (N, C, Samples, 1)
X_val   = np.expand_dims(X_val,   axis=-1)

# 2) One-hot encode labels
y_train_cat = to_categorical(y_train, num_classes=2)
y_val_cat   = to_categorical(y_val,   num_classes=2)

# 3) Build EEGNet
Chans   = X_train.shape[1]  # C
Samples = X_train.shape[2]  # window_size
model = EEGNet(
    nb_classes   = 2,
    Chans        = Chans,
    Samples      = Samples,
    dropoutRate  = 0.5,
    kernLength   = 250,   # half your sampling rate if it's 500
    F1           = 8,
    D            = 2,
    F2           = 16,
    norm_rate    = 0.25,
    dropoutType  = 'Dropout'
)

model.compile(loss='categorical_crossentropy',
              optimizer=Adam(lr=1e-3),
              metrics=['accuracy'])

# 4) Fit
history = model.fit(
    X_train, y_train_cat,
    batch_size=16,
    epochs=20,
    validation_data=(X_val, y_val_cat)
)

# 5) Save
model.save("eegnet_all_subjects_all_sessions.h5")








import numpy as np
import json
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from Classification.PhasesClassification.EEGModels import EEGNet

# ---------------------------
# Padding function (global)
# ---------------------------
def pad_trials(trials):
    max_time = max(trial.shape[0] for trial in trials)
    padded_trials = []
    for trial in trials:
        pad_width = max_time - trial.shape[0]
        if pad_width > 0:
            trial_padded = np.pad(trial, pad_width=((0, pad_width), (0, 0)), mode='constant')
        else:
            trial_padded = trial
        padded_trials.append(trial_padded)
    return np.array(padded_trials)

# ---------------------------
# Load data from a single JSON file
# ---------------------------
def load_data_from_json(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    trials = []
    weight_labels = []
    texture_labels = []
    for trial in data["experiments"]:
        eeg_trial = np.array(trial["eeg"])  # shape: (time, channels)
        trials.append(eeg_trial)
        weight_labels.append(trial["experimental_conditions"]["weight_id"])
        texture_labels.append(trial["experimental_conditions"]["surface_id"])

    X = np.array(trials, dtype=object)  # ragged array, doesn't require dimensions to match
    y_weight = np.array(weight_labels)
    y_texture = np.array(texture_labels)

    return X, y_weight, y_texture

# ---------------------------
# Load data from all participants and sessions
# ---------------------------
def load_all_data(participants, sessions):
    X_list = []
    yw_list = []
    yt_list = []

    for p in participants:
        for s in sessions:
            file_path = f"/Users/giovanninocerino/Desktop/Associations/EEG Research/math_file/P{p}/WS_P{p}_S{s}.json"
            print(f"Loading data from WS_P{p}_S{s}")
            X_curr, yw_curr, yt_curr = load_data_from_json(file_path)
            X_list.append(X_curr)
            yw_list.append(yw_curr)
            yt_list.append(yt_curr)

    return X_list, yw_list, yt_list

# ---------------------------
# Main pipeline
# ---------------------------
participants = range(1, 4)  # 1..12
sessions = range(1, 4)  # 1..9

# 1. Load data (ragged)
X_list, y_weight_list, y_texture_list = load_all_data(participants, sessions)

# 2. Flatten
all_trials = np.concatenate(X_list, axis=0)  # shape: (N,), each entry (Ti, C)
y_weight_all = np.concatenate(y_weight_list, axis=0)
y_texture_all = np.concatenate(y_texture_list, axis=0)

print(f"Total trials loaded: {len(all_trials)}")
print(f"Weight labels shape: {y_weight_all.shape}")
print(f"Texture labels shape: {y_texture_all.shape}")

# 3. Pad
X_padded = pad_trials(all_trials)  # shape: (N, T_max, C)
print(f"X_padded shape (after padding): {X_padded.shape}")
X_padded = np.transpose(X_padded, (0, 2, 1))  # (N, C, T)
print(f"X_transposed shape (C first): {X_padded.shape}")
X_padded = np.expand_dims(X_padded, axis=-1)  # (N, C, T, 1)
print(f"X_final shape (4D): {X_padded.shape}")

# 4. Labels to one-hot
# Get unique weights (e.g. ['330g', '660g', '990g', '1320g'])
y_weight_all=y_weight_all.flatten()
unique_weights = sorted(set(y_weight_all.tolist()))
weight_to_id = {w: i for i, w in enumerate(unique_weights)}
# Map all string labels to integers
y_weight_all_int = np.array([weight_to_id[w] for w in y_weight_all.tolist()])
y_all = to_categorical(y_weight_all_int, num_classes=4)

# 5. Split
X_train, X_val, y_train, y_val = train_test_split(X_padded, y_all, test_size=0.2, random_state=42)
print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
print(f"X_val shape: {X_val.shape}, y_val shape: {y_val.shape}")

# 6. Build + train model
Chans = X_train.shape[1]
Samples = X_train.shape[2]

model = EEGNet(nb_classes=4, Chans=Chans, Samples=Samples, dropoutRate=0.5)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_val, y_val))

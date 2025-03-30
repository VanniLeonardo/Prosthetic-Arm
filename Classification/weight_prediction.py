import numpy as np
import json
import mne
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from EEGModels import EEGNet, ShallowConvNet, DeepConvNet

def pad_trials(trials):
    # Determine the maximum number of time points across all trials
    max_time = max(trial.shape[0] for trial in trials)
    padded_trials = []
    for trial in trials:
        n_time, n_channels = trial.shape
        pad_width = max_time - n_time
        if pad_width > 0:
            # Pad along the time dimension (first dimension)
            trial_padded = np.pad(trial, pad_width=((0, pad_width), (0, 0)), mode='constant', constant_values=0)
        else:
            trial_padded = trial
        padded_trials.append(trial_padded)
    return np.array(padded_trials)

def pad_all_trials(trials):
    # Determine the maximum number of time points across all trials
    max_time = max(trial.shape[1] for trial in trials)
    padded_trials = []
    for trial in trials:
        n_experiment, n_time, n_channels  = trial.shape
        pad_width = max_time - n_time
        if pad_width > 0:
            # Pad along the time dimension (first dimension)
            trial_padded = np.pad(trial, pad_width=((0, pad_width), (0, 0)), mode='constant', constant_values=0)
        else:
            trial_padded = trial
        padded_trials.append(trial_padded)
    return np.array(padded_trials)


def load_data_from_json(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    trials = []
    weight_labels = []   # For weight: 4 classes
    texture_labels = []  # For texture: 2 classes
    for trial in data["experiments"]:
        eeg_trial = np.array(trial["eeg"]) # shape is (n_timepoints, n_channels) (4000, 32)
        trials.append(eeg_trial) # size is 34
        weight_labels.append(trial["experimental_conditions"]["weight_id"])  
        texture_labels.append(trial["experimental_conditions"]["surface_id"])
    # Now 'trials' is a list of 34 arrays

    padded_trials = pad_trials(trials)  # list of 34 arrays each with the same number of timepoints
    X = np.array(padded_trials)  # shape: (n_trials, n_channels, n_timepoints)
    # Why is X not (trials, timepoints, channels)?

    y_weight = np.array(weight_labels)
    y_texture = np.array(texture_labels)
    # print the shape of the data
    print(f"X shape: {X.shape}")
    print(f"y_weight shape: {y_weight.shape}")
    print(f"y_texture shape: {y_texture.shape}")
    return X, y_weight, y_texture

def load_all_data():

    X_list = []
    yw_list = []
    yt_list = []

    # Suppose your 12 participants are labeled P1..P12 
    for p in range(1, 3):          # 1..12
        # Each participant has 9 WS files labeled from S1..S9
        for s in range(1, 2):      # 1..9
            file_name = f"/Users/giovanninocerino/Desktop/Associations/EEG Research/math_file/P{p}/WS_P{p}_S{s}.json"
            print(f"loading data from  WS_P{p}_S{s} time")
            X_curr, yw_curr, yt_curr = load_data_from_json(file_name)
            X_list.append(X_curr)
            yw_list.append(yw_curr)
            yt_list.append(yt_curr)
    return X_list, yw_list, yt_list

X_all, y_weight_all, y_texture_all = load_all_data()


X_pad = pad_trials(X_all)
# Now concatenate along the "trial" dimension (axis=0)
X_fin = np.concatenate(X_pad, axis=0)
y_weight_all = np.concatenate(y_weight_all, axis=0)
y_texture_all = np.concatenate(y_texture_all, axis=0)

for trial in X_fin:
   print(trial.shape)

   


print("Combined dataset shape:", X_all.shape)
print("Weight labels shape:", y_weight_all.shape)
print("Texture labels shape:", y_texture_all.shape)

# We should combine weight and texture labels into a single label
# STILL TO BE DONE
y_all = to_categorical(y_weight_all)  # one-hot encoding

X_train, X_val, y_train, y_val = train_test_split(X_all, y_all, test_size=0.2, random_state=42)


model = EEGNet(nb_classes=4, Chans=32, Samples=4769, dropoutRate=0.5)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(
    X_train,    # shape: (n_trials, Chans, Samples, 1)
    y_train,    # shape: (n_trials,) or (n_trials, nb_classes) if one-hot
    epochs=50,
    batch_size=32,
    validation_data=(X_val, y_val)
)


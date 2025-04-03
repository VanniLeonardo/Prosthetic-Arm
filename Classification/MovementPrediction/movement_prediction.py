import os
import numpy as np
import orjson
import json
import pickle
from EEGModels import EEGNet
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import StratifiedKFold
import tensorflow as tf
import matplotlib.pyplot as plt

# Global sampling rate
s_r = 500  # Hz

# ---------------------
# RAW DATA EXTRACTION FUNCTIONS
# ---------------------
def extract_window_movements(p, s):
    filepath = f"/Users/giovanninocerino/Desktop/Associations/EEG Research/math_file/P{p}/WS_P{p}_S{s}.json"
    with open(filepath, "rb") as f:
        data = json.load(f)
    time_spans = []
    for trial in data["experiments"]:
        start = int(trial["timestamps"]["trial_start"][0][0] * s_r) - 999
        end   = int(trial["timestamps"]["trial_end"][0][0] * s_r) - 999
        time_spans.append([start, end])
    return np.array(time_spans)

def extract_hs(p, s):
    filepath = f"/Users/giovanninocerino/Desktop/Associations/EEG Research/math_file/P{p}/HS_P{p}_S{s}.json"
    with open(filepath, "rb") as f:
        data = json.load(f)
    eeg_signal = np.array(data["EEG"]["data"])  # Expected shape: (T, C) [time x channels]
    return eeg_signal

def assign_labels(p, s):
    eeg = extract_hs(p, s)
    labels = np.zeros(len(eeg), dtype=int)
    time_spans = extract_window_movements(p, s)
    for start, end in time_spans:
        labels[start:end] = 1  
    return labels

def gather_raw_data(participants, sessions):
    """
    Gathers raw continuous EEG data and sample-level labels (without epoching)
    from all specified participants and sessions.
    Returns two lists: one list of raw EEG arrays and one list of corresponding label arrays.
    """
    raw_X_list = []
    raw_y_list = []
    for p in participants:
        for s in sessions:
            print(f"Gathering raw data for Participant {p}, Session {s}")
            eeg = extract_hs(p, s)      # (T, C)
            labels = assign_labels(p, s)  # (T,)
            raw_X_list.append(eeg)
            raw_y_list.append(labels)
    return raw_X_list, raw_y_list

# Define file names for saving raw continuous data using pickle
raw_X_file = "raw_X_all.pkl"
raw_y_file = "raw_y_all.pkl"

# Define participants and sessions (adjust as needed)
participants = [1,2,3,4,5,6,7,8,9,10,11,12]  # For testing; expand later as needed.
sessions = [1, 2, 3,4,5,6,7,8,9]   # For testing; adjust as needed.

# If raw data pickle files exist, load them; otherwise, extract and save raw data.
if os.path.exists(f"Classification/MovementPrediction/{raw_X_file}") and os.path.exists(f"Classification/MovementPrediction/{raw_y_file}"):
    print("Loading raw continuous data from pickle files...")
    with open(f"Classification/MovementPrediction/{raw_X_file}", "rb") as f:
        raw_X_all = pickle.load(f)
    with open(f"Classification/MovementPrediction/{raw_y_file}", "rb") as f:
        raw_y_all = pickle.load(f)
else:
    print("Extracting raw continuous data and saving to pickle files...")
    raw_X_all, raw_y_all = gather_raw_data(participants, sessions)
    with open(raw_X_file, "wb") as f:
        pickle.dump(raw_X_all, f)
    with open(raw_y_file, "wb") as f:
        pickle.dump(raw_y_all, f)

# ---------------------
# EPOCHING FUNCTIONS
# ---------------------
def create_epochs(eeg, labels, window_size=500, step=250, threshold=0.5):
    """
    Splits continuous EEG (T x C) and corresponding labels (T,) into epochs.
    Each epoch is of length 'window_size' and is labeled as movement (1) if the fraction
    of movement samples is >= threshold.
    """
    T, channels = eeg.shape  # (time, channels)
    X_list = []
    y_list = []
    for start_idx in range(0, T - window_size, step):
        end_idx = start_idx + window_size
        window_data = eeg[start_idx:end_idx, :]      # shape: (window_size, channels)
        window_labels = labels[start_idx:end_idx]      # shape: (window_size,)
        frac_movement = np.mean(window_labels)
        window_label = 1 if frac_movement >= threshold else 0
        # Transpose to get shape (channels, window_size) for EEGNet
        window_data = window_data.T
        X_list.append(window_data)
        y_list.append(window_label)
    X = np.array(X_list)  # (num_epochs, channels, window_size)
    y = np.array(y_list)
    return X, y

def epoch_raw_data(raw_X_list, raw_y_list, window_size=500, step=250, threshold=0.5):
    """
    Processes a list of raw EEG sessions and label arrays.
    Applies epoching to each session and concatenates the results.
    """
    X_list = []
    y_list = []
    for eeg, labels in zip(raw_X_list, raw_y_list):
        X, y = create_epochs(eeg, labels, window_size, step, threshold)
        X_list.append(X)
        y_list.append(y)
    X_all = np.concatenate(X_list, axis=0)
    y_all = np.concatenate(y_list, axis=0)
    return X_all, y_all

# Define file names for epoched data
epoched_X_file = "X_all.npy"
epoched_y_file = "y_all.npy"

if os.path.exists(f"Classification/MovementPrediction/{epoched_X_file}") and os.path.exists(f"Classification/MovementPrediction/{epoched_y_file}"):
    print("Loading epoched data from npy files...")
    X_all = np.load(f"Classification/MovementPrediction/{epoched_X_file}")
    y_all = np.load(f"Classification/MovementPrediction/{epoched_y_file}")
else:
    print("Creating epochs from raw data and saving to npy files...")
    X_all, y_all = epoch_raw_data(raw_X_all, raw_y_all, window_size=500, step=250, threshold=0.5)
    np.save(epoched_X_file, X_all)
    np.save(epoched_y_file, y_all)

# Prepare data for EEGNet: Add final singleton dimension (N, channels, window_size, 1)
X_all = np.expand_dims(X_all, axis=-1)
print(f"Final epoched data shape: {X_all.shape}, Labels shape: {y_all.shape}")

# ---------------------
# CROSS-VALIDATION TRAINING
# ---------------------
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
val_accuracies = []

for fold, (train_idx, val_idx) in enumerate(skf.split(X_all, y_all)):
    print(f"Starting fold {fold + 1}")
    X_train, X_val = X_all[train_idx], X_all[val_idx]
    y_train, y_val = y_all[train_idx], y_all[val_idx]
    y_train_cat = to_categorical(y_train, num_classes=2)
    y_val_cat = to_categorical(y_val, num_classes=2)
    
    # Clear session to avoid interference between folds
    tf.keras.backend.clear_session()
    
    Chans = X_train.shape[1]
    Samples = X_train.shape[2]
    model = EEGNet(
        nb_classes = 2,
        Chans = Chans,
        Samples = Samples,
        dropoutRate = 0.5,
        kernLength = 250,  # half of 500 Hz
        F1 = 8,
        D = 2,
        F2 = 16,
        norm_rate = 0.25,
        dropoutType = 'Dropout'
    )
    
    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(learning_rate=1e-3),
                  metrics=['accuracy'])
    
    history = model.fit(
        X_train, y_train_cat,
        epochs=20,
        batch_size=16,
        validation_data=(X_val, y_val_cat),
        verbose=1
    )
    
    loss, acc = model.evaluate(X_val, y_val_cat, verbose=0)
    print(f"Fold {fold+1} validation accuracy: {acc:.4f}")
    val_accuracies.append(acc)

print("Average cross-validation accuracy:", np.mean(val_accuracies))

# Optionally, plot the training history of the last fold:
plt.figure(figsize=(10, 5))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Last Fold Loss")
plt.legend()
plt.show()

plt.figure(figsize=(10, 5))
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Last Fold Accuracy")
plt.legend()
plt.show()

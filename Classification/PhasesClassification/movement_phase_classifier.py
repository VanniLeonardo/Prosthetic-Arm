import os
import glob
import json
import numpy as np
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import StratifiedKFold, train_test_split
import matplotlib.pyplot as plt
from EEGModels import EEGNet
from tensorflow.keras.optimizers import Adam

s_r = 500  # sampling rate

def extract_labels(all_lift_path, session_path, s_r=500):
    with open(all_lift_path, "rb") as f:
        data = json.load(f)

    columns = data["columns"]
    trials = [dict(zip(columns, row)) for row in data["data"]]
    session = np.load(session_path, allow_pickle=True)
    T = session.shape[1]
    S = next((i for i in range(10) if f"S{i}" in session_path), None)
    labels = np.zeros(T, dtype=int)

    for trial in trials:
        if trial["Run"] != S:
            continue
        try:
            shift_sec = 2
            start_sample = int((trial["StartTime"] - shift_sec) * s_r)
            stop_sample = int((trial["StartTime"] + trial["LEDOff"] + shift_sec) * s_r)
            next_trials = [t for t in trials if t["Run"] == S and t["Lift"] > trial["Lift"]]
            if next_trials:
                next_start_sample = int((next_trials[0]["StartTime"] - shift_sec) * s_r)
                stop_sample = min(stop_sample, next_start_sample)
            start_sample = max(0, start_sample)
            stop_sample = min(T, stop_sample)

            hand_start = start_sample + int(trial["tHandStart"] * s_r)
            both_touch = start_sample + int(trial["tBothDigitTouch"] * s_r)
            lift_off = start_sample + int(trial["tLiftOff"] * s_r)
            replace = start_sample + int(trial["tReplace"] * s_r)
            both_released = start_sample + int(trial["tBothReleased"] * s_r)
            hand_stop = start_sample + int(trial["tHandStop"] * s_r)

            for t in range(start_sample, stop_sample):
                if t < hand_start:
                    labels[t] = 0
                elif t < both_touch:
                    labels[t] = 1
                elif t < lift_off:
                    labels[t] = 2
                elif t < replace:
                    labels[t] = 3
                elif t < both_released:
                    labels[t] = 4
                elif t < hand_stop:
                    labels[t] = 5
        except:
            continue
    return labels

def load_all_sessions(number):
    base_path = "/Users/giovanninocerino/Desktop/Associations/EEG Research/Dataset"
    user_folder = os.path.join(base_path, f"P{number}")
    all_lift_path = os.path.join(user_folder, f"P{number}_AllLifts.json")

    if not os.path.exists(all_lift_path):
        raise FileNotFoundError(f"All lift file not found: {all_lift_path}")

    session_pattern = os.path.join(user_folder, f"HS_P{number}_S*_eeg.npy")
    session_paths = sorted(glob.glob(session_pattern))

    if not session_paths:
        raise FileNotFoundError(f"No session file found for user P{number} using pattern: {session_pattern}")

    all_labels = []
    all_sessions = []

    for session_path in session_paths:
        print("Processing session file:", session_path)
        labels = extract_labels(all_lift_path, session_path)
        session = np.load(session_path, allow_pickle=True)
        all_labels.append(labels)
        all_sessions.append(session)

    return all_labels, all_sessions

def get_all_data():
    all_data = []
    all_labels = []
    for number in range(1, 13):
        try:
            labels, sessions = load_all_sessions(number)
            all_data.extend(sessions)
            all_labels.extend(labels)
        except Exception as e:
            print(f"Error loading data for user {number}: {e}")
    return all_data, all_labels

def create_phase_epochs(eeg_list, label_list, window_size=500, step=250):
    X_all, y_all = [], []
    for eeg, labels in zip(eeg_list, label_list):
        T = eeg.shape[1]
        for start in range(0, T - window_size, step):
            end = start + window_size
            window_data = eeg[:, start:end]
            window_labels = labels[start:end]
            if np.any(window_labels == -1):
                continue
            label = np.bincount(window_labels).argmax()
            X_all.append(window_data)
            y_all.append(label)
    return np.array(X_all), np.array(y_all)

def train_model(X, y, n_classes, n_splits=5):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    models = []
    histories = []
    for train_idx, val_idx in skf.split(X, y):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = to_categorical(y[train_idx], n_classes), to_categorical(y[val_idx], n_classes)
        model = EEGNet(nb_classes=n_classes, Chans=X.shape[1], Samples=X.shape[2], dropoutRate=0.5,
                       kernLength=250, F1=8, D=2, F2=16, dropoutType='Dropout', norm_rate=0.25)
        model.compile(optimizer=Adam(0.001), loss='categorical_crossentropy', metrics=['accuracy'])
        history = model.fit(X_train, y_train, epochs=10, batch_size=16, validation_data=(X_val, y_val))
        models.append(model)
        histories.append(history)
    return models, histories

def evaluate_model(model, X_test, y_test):
    return model.evaluate(X_test, y_test, verbose=1)

def plot_history(history):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train')
    plt.plot(history.history['val_accuracy'], label='Val')
    plt.title('Accuracy')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train')
    plt.plot(history.history['val_loss'], label='Val')
    plt.title('Loss')
    plt.legend()
    plt.tight_layout()
    plt.show()

def main():
    all_data, all_labels = get_all_data()
    X, y = create_phase_epochs(all_data, all_labels)
    n_classes = len(np.unique(y))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)
    models, histories = train_model(X_train, y_train, n_classes)

    for i, model in enumerate(models):
        print(f"Evaluating model {i + 1}")
        evaluate_model(model, X_test, to_categorical(y_test, n_classes))
        plot_history(histories[i])

if __name__ == "__main__":
    main()
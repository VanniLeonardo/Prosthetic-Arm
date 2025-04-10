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
# EXTRACTING LABELS
# ---------------------
def extract_labels(all_lift_path, session_path, s_r=500):
    PHASE_LABELS = {
        "rest": 0, "reach": 1, "preload": 2,
        "lift": 3, "release": 4, "retract": 5
    }

    with open(all_lift_path, "rb") as f:
        data = json.load(f)

    columns = data["columns"]
    trials = [dict(zip(columns, row)) for row in data["data"]]

    session = np.load(session_path, allow_pickle=True)
    T = session.shape[1]

    S = next((i for i in range(10) if f"S{i}" in session_path), None)
    if S is None:
        raise ValueError("Could not determine session number from path.")
    
    print(S,T)
    labels = np.zeros(T, dtype=int)

    valid_trial_count = 0

    for trial in trials:
        if trial["Run"] != S:
            continue

        try:
            shift_sec = 2
            start_sample = int((trial["StartTime"]- shift_sec) * s_r)
            stop_sample = int((trial["StartTime"] + trial["LEDOff"] + 3) * s_r)

            # Truncate to avoid overlap with next trial
            next_trials = [t for t in trials if t["Run"] == S and t["Lift"] > trial["Lift"]]
            if next_trials:
                next_start_sample = int((next_trials[0]["StartTime"] - shift_sec) * s_r)
                stop_sample = min(stop_sample, next_start_sample)

            # Clip within bounds
            start_sample = max(0, start_sample)
            stop_sample = min(T, stop_sample)

            # Calculate all event sample indices
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

        except (KeyError, TypeError, ValueError) as e:
            print(f"🚫 Skipping trial {trial.get('Lift', '?')} due to error: {e}")
            continue
    transitions = np.sum((labels[:-1] == 5) & (labels[1:] == 0))
    print(f"Phase transitions (5 → 0): {transitions}")

    return labels

#trial
print(extract_labels("/Users/giovanninocerino/Desktop/Associations/EEG Research/math_file/P1/P1_AllLifts.json","/Users/giovanninocerino/Desktop/Associations/EEG Research/math_file/P2/HS_P1_S1_eeg.npy", s_r=500))

# ---------------------
# LOADING DATA
# ---------------------
def load_person(person_path, s_r=500):
    person_data = 
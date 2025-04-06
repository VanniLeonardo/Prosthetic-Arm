import os
import numpy as np
from tensorflow.keras.models import load_model

# === SETTINGS ===
model_path = "EEGNet_BestHPO_FinalModel.keras"
base_path = "/Users/giovanninocerino/Desktop/Associations/EEG Research/math_file"
window_size = 500
step = 250

# === Load the trained model ===
model = load_model(model_path)
print("[✓] Model loaded.")

# === Epoching without labels ===
def create_epochs_unlabeled(eeg, window_size=500, step=250):
    T, C = eeg.shape
    X_list = []
    for start_idx in range(0, T - window_size, step):
        end_idx = start_idx + window_size
        window_data = eeg[start_idx:end_idx, :]  # (window_size, C)
        window_data = window_data.T  # (C, window_size) for EEGNet
        X_list.append(window_data)
    return np.array(X_list)  # (N, C, Samples)

# === Run predictions for each participant ===
for p in range(1, 13):
    eeg_path = os.path.join(base_path, f"P{p}", f"HS_P{p}_S5_eeg.npy")
    if not os.path.exists(eeg_path):
        print(f"[!] Missing EEG file for P{p}")
        continue

    eeg = np.load(eeg_path)  # shape: (T, C)
    X = create_epochs_unlabeled(eeg, window_size, step)
    X = np.expand_dims(X, axis=-1)  # (N, C, Samples, 1)

    # Predict
    predictions = model.predict(X, verbose=0)  # shape: (N, 2)
    predicted_classes = np.argmax(predictions, axis=1)

    print(f"\n[P{p}] Predicted {len(predicted_classes)} windows.")
    print("Predicted class distribution:", np.bincount(predicted_classes))
    
    # Optionally: save predictions
    out_path = os.path("/Users/giovanninocerino/Prosthetic-Arm/Classification/MovementPrediction/Predictions")
    np.save(out_path, predicted_classes)
    print(f"[✓] Saved predictions to {out_path}")

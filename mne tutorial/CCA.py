'''CCA in EEG Artifact Removal'''

import mne
import numpy as np
from sklearn.cross_decomposition import CCA
import matplotlib.pyplot as plt

# Step 1: Load Physionet EEG Motor Imagery Dataset
print("Fetching and loading the Physionet Motor Imagery dataset...")
runs = [3, 7, 11]  # Right and left-hand motor movement
subject = 1  # Change subject ID if needed

# Download and load the raw data
raw_files = mne.datasets.eegbci.load_data(subject=subject, runs=runs)
raw = mne.io.concatenate_raws([mne.io.read_raw_edf(f, preload=True) for f in raw_files])
raw.rename_channels(lambda x: x.strip('.'))



# Step 2: Preprocess the Data
print("Preprocessing data...")
# Remove power-line noise (60 Hz in the US where the data was collected)
raw.notch_filter(freqs=[60])
# Bandpass filter (1-40 Hz)
raw.filter(1., 40., fir_design='firwin', verbose=False)
# Pick only EEG channels
raw.pick_types(eeg=True, exclude='bads')

# Downsample to reduce computation
raw.resample(128, npad="auto")

# Step 3: Annotate Events and Extract Epochs
print("Extracting epochs...")
events, event_id = mne.events_from_annotations(raw)  # {1: 45, 2: 21, 3: 24}

# Use both left (T1) and right (T2) hand motor imagery events
event_id = {'Left Hand': 2, 'Right Hand': 3} # In case of runs [6, 10, 14] here we have fists and feet

# Create epochs (e.g., -1 to 2 seconds around the events)
epochs = mne.Epochs(raw, events, event_id, tmin=-1.0, tmax=2.0, baseline=(None, 0), preload=True)
data = epochs.get_data()  # Shape: (n_epochs, n_channels, n_times)



# REMOVE REST STATE (T0=1) FROM RAW DATA
# Filter events for T1 and T2
filtered_events = events[events[:, 2] != 1]  # Exclude T0 events

# Create a mask for time points corresponding to T1 and T2 epochs
sfreq = raw.info['sfreq']  # Sampling frequency
raw_data = raw.get_data()  # Shape: (n_channels, n_times)
n_times = raw_data.shape[1]

# Initialize mask (False for all time points)
mask = np.zeros(n_times, dtype=bool)

# Iterate over T1 and T2 events to include their time ranges
tmin, tmax = -1.0, 2.0  # Time window around events
for event in filtered_events:
    start_sample = int(event[0] + tmin * sfreq)
    end_sample = int(event[0] + tmax * sfreq) + 1  # Add 1 sample to include endpoint
    mask[start_sample:end_sample] = True  # Include T1 and T2 time points

# Apply the mask to raw data
matched_raw_data = raw_data[:, mask]  # Keep only T1 and T2 epoch samples

# Create a new Raw object with the filtered data
from mne.io import RawArray
matched_times = raw.times[mask]  # Adjust time points to match filtered data
matched_raw = RawArray(matched_raw_data, raw.info)


# Step 4: Prepare Data for CCA
print("Preparing data for CCA...")
# Flatten epochs for CCA (samples x features)
n_epochs, n_channels, n_times = data.shape
data_flat = data.reshape(n_epochs, -1)  # Shape: (n_epochs, n_channels * n_times)

# Create a time-delayed version of the data
data_delayed = np.roll(data_flat, shift=-1, axis=0)  # Simple time delay

# Step 5: Apply CCA
print("Applying CCA for artifact removal...")
# CCA requires that the number of components does not exceed the smallest dimension of the input data
# Otherwise the system is overdetermined and cannot be solved
cca = CCA(n_components=min(data_flat.shape[0], data_flat.shape[1]))
# where data_flat.shape[0] corresponds to the number of epochs,
# data_flat.shape[1] corresponds to n_channels * n_times

# Find linear transformations of the two datasets that maximize their mutual correlation
cca.fit(data_flat, data_delayed) # handles normalization internally 

# Transform the data, i.e.,
# project the original data into a new space defined by the linear combinations of features
data_transformed, data_delayed_transformed = cca.transform(data_flat, data_delayed)
# The data_transformed array contains the canonical components that represent linear combinations of the original signals
# Each column corresponds to one component

'''
print(f"Mean of data_transformed: {data_transformed.mean(axis=0)}")
print(f"Std of data_transformed: {data_transformed.std(axis=0)}")

normalized_components = (data_transformed - data_transformed.mean(axis=0)) / data_transformed.std(axis=0)

print(f"Mean of normalized_components: {normalized_components.mean(axis=0)}")
print(f"Std of normalized_components: {normalized_components.std(axis=0)}")
'''

# Step 6: Identify and Remove Artifact Components
# For each component in data_transformed, calculate its autocorrelation
def autocorrelation(signal):
    return np.corrcoef(signal[:-1], signal[1:])[0, 1]  # internally normalizes the signals
autocorr_values = [autocorrelation(data_transformed[:, i]) for i in range(data_transformed.shape[1])]
# Components with low autocorrelation are treated as artifacts
threshold = 0.05  # Adjust the threshold as needed - higher threshold is more aggressive and tends to remove meaningful signal
artifact_indices = [i for i, ac in enumerate(autocorr_values) if -threshold < ac < threshold]

print("Artifact components identified:", artifact_indices)

# Zero out artifact components
data_clean_transformed = data_transformed.copy()
data_clean_transformed[:, artifact_indices] = 0


# Step 7: Reconstruct Cleaned Signals
data_clean_flat_tuple = cca.inverse_transform(data_clean_transformed, data_delayed_transformed)

# Handle tuple output
if isinstance(data_clean_flat_tuple, tuple):
    data_clean_flat = data_clean_flat_tuple[0]  # Use the first view
else:
    data_clean_flat = data_clean_flat_tuple  # In case it's a single array

# Reshape to the original dimensions
data_clean = data_clean_flat.reshape(n_epochs, n_channels, n_times)



# List all channel names
channel_names = raw.info['ch_names']
# Get all channel indices
channel_indices = list(range(len(channel_names)))
print("Channel Names:", channel_names)
print("Channel Indices:", channel_indices)



# Step 8: Visualize Results for Left and Right Hands Tasks
print("Visualizing results...")

'''
# RAW VS CLEANED DATA EXACT COMPARISON
from mne.io import RawArray

# Create an MNE Info object using the original raw data
info = raw.info

# Convert the cleaned data (assumed to be in shape (n_channels, n_times)) into a RawArray
data_clean_transposed = data_clean.transpose(1, 0, 2).reshape(len(raw.ch_names), -1)  # (n_channels, n_times)
raw_clean = RawArray(data_clean_transposed, info)

# Plot filtered data (raw data without the rest state)
matched_raw.plot(picks=['C3', 'Cz', 'C4'], scalings={'eeg': 20e-6}, title="Raw Data Without T0 Events")

# Plot cleaned data
raw_clean.plot(picks=['C3', 'Cz', 'C4'], scalings={'eeg': 20e-6}, title='Cleaned EEG Data', show=True, block=True)
# add argument picks=['C3', 'Cz', 'C4'] to select specific channels to plot
'''


index_epoch = 4  # Change to view different epochs # Eye blinks in epochs with indices 4 and 9

# Define the channels of interest
channels_of_interest = ['F8', 'Fp1', 'Af4']  # 'C3', 'Cz', 'C4', 'T8', 'T7', 'Fcz', 'Afz', 'Fp1', 'Af4'

# Get indices for channels of interest
channel_indices = [epochs.ch_names.index(ch) for ch in channels_of_interest]

# Map the selected epoch to its corresponding event ID
event_idx = epochs.selection[index_epoch]  # Get the event index
plotted_event_id = events[event_idx, 2]  # Get the event ID (third column)
    
# Map the event ID to the task name
task = {v: k for k, v in event_id.items()}[plotted_event_id]
print(f"The epoch {index_epoch+1} corresponds to: {task}")

# Plot raw vs clean data for a specifc epoch for channels_of_interest
fig, axes = plt.subplots(len(channels_of_interest), 1, figsize=(10, 3 * len(channels_of_interest)))
for i, ch_idx in enumerate(channel_indices):
    ch = channels_of_interest[i]  # Get the channel name for labeling
    axes[i].plot(epochs.times, data[index_epoch, ch_idx, :]* 1e6, label=f"Raw: {ch}", alpha=0.7)
    axes[i].plot(epochs.times, data_clean[index_epoch, ch_idx, :]* 1e6, label=f"Cleaned: {ch}", alpha=0.7)
    axes[i].set_title(f"{task} Motor Imagery: {ch}. Epoch:{index_epoch+1}")
    axes[i].set_xlabel("Time (s)")
    axes[i].set_ylabel("Amplitude (µV)")
    axes[i].legend()
plt.tight_layout()
plt.show()

'''
# Plot each separately
for i, ch_idx in enumerate(channel_indices):
    plt.figure(figsize=(10, 6))
    time = epochs.times  # Time in seconds for each sample
    ch = channels_of_interest[i]  # Get the channel name for labeling
    plt.plot(time, data[index_epoch, ch_idx, :]* 1e6, label=f'Raw Signal (Channel {ch}, Epoch {index_epoch+1})', alpha=0.7)
    plt.plot(time, data_clean[index_epoch, ch_idx, :]* 1e6, label=f'Cleaned Signal (Channel {ch}, Epoch {index_epoch+1})', alpha=0.7)
    plt.title(f"Raw vs Cleaned Signal (Channel {ch}, Epoch {index_epoch+1}, Task: {task})")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude (µV)")
    plt.legend()
    plt.show()



# Plot the PSD for short time intervals to inspect signal frequency at spikes
# Useful in identifying types of artifacts as well as in distinguishing between a genuine neural signal and an artifact
# Spikes at low frequencies (0-4 Hz) tend to be ocular artifacts
# Spikes at above 30 Hz are likely muscular artifacts

# Get the sampling frequency
sfreq = epochs.info['sfreq']  # Hz

# Extract the data for a specified epoch
epoch_data = epochs.get_data()[index_epoch]  # Shape: (n_channels, n_times)

# Select the channel with the spike, e.g., 'T8'
pick_channel = 'T8'  # Replace 'T8' if needed
channel_index = epochs.ch_names.index(pick_channel)
spike_data_full = epoch_data[channel_index, :]  # Full data for the channel

start = 0.63
end = 0.76

# Define time window of the spike
start_sample = int(start * sfreq)
end_sample = int(end * sfreq)
spike_data = spike_data_full[start_sample:end_sample]  # Extract the spike

from scipy.signal import welch
import matplotlib.pyplot as plt

# Compute PSD for the spike data
frequencies, psd = welch(spike_data, fs=sfreq, nperseg=128)  # Adjust nperseg for resolution

# Extract the cleaned data for a specified epoch
cleaned_epoch_data = data_clean[index_epoch]  # Shape: (n_channels, n_times)

# Select the channel with the spike
spike_data_clean_full = cleaned_epoch_data[channel_index, :]  # Full cleaned data for the channel
spike_data_clean = spike_data_clean_full[start_sample:end_sample]  # Extract the spike

# Compute PSD for the cleaned spike data
frequencies_clean, psd_clean = welch(spike_data_clean, fs=sfreq, nperseg=128)

# Plot both raw and cleaned PSD
plt.figure(figsize=(10, 6))
plt.semilogy(frequencies, psd, label=f"Raw Spike ({start}–{end} s)", color='blue')
plt.semilogy(frequencies_clean, psd_clean, label=f"Cleaned Spike ({start}–{end} s)", color='red')
plt.axvspan(0, 4, color='red', alpha=0.2, label='Delta (0-4 Hz)')
plt.axvspan(30, 50, color='orange', alpha=0.2, label='Muscle Artifacts (30-50 Hz)')
plt.xlabel("Frequency (Hz)")
plt.ylabel("Power Spectral Density (µV²/Hz)")
plt.title(f"PSD Comparison: Raw vs Cleaned (Spike in {pick_channel}, {start}–{end} s)")
plt.legend()
plt.show()



# Get channel names and their indices
channel_names = epochs.info['ch_names']
motor_channels = ['C3', 'Cz', 'C4']
motor_indices = [channel_names.index(ch) for ch in motor_channels]

# Plot raw and cleaned signals for each motor channel across all samples
import matplotlib.pyplot as plt

for idx, channel in zip(motor_indices, motor_channels):
    raw_signal = epochs.get_data()[:, idx, :].flatten()
    clean_signal = data_clean[:, idx, :].flatten()

    plt.figure(figsize=(10, 6))
    plt.plot(raw_signal, label='Raw Signal', alpha=0.7)
    plt.plot(clean_signal, label='Cleaned Signal', alpha=0.7)
    plt.title(f'Raw vs Cleaned Signal ({channel})')
    plt.xlabel('Time (samples)')
    plt.ylabel('Amplitude (V)')
    plt.legend()
    plt.show()


# Plot PSD wrt frequency -> Check that the signal in Mu and Beta bands is preserved
from scipy.signal import welch

# Convert PSD to µV²/Hz and use a logarithmic y-axis
for idx, channel in zip(motor_indices, motor_channels):
    # Compute PSD for raw and cleaned data
    f_raw, psd_raw = welch(epochs.get_data()[:, idx, :].flatten(), fs=raw.info['sfreq'], nperseg=1024)
    f_clean, psd_clean = welch(data_clean[:, idx, :].flatten(), fs=raw.info['sfreq'], nperseg=1024)

    # Convert PSD to µV²/Hz
    psd_raw_microvolts = psd_raw * 1e12
    psd_clean_microvolts = psd_clean * 1e12

    # Plot PSD
    plt.figure(figsize=(10, 6))
    plt.semilogy(f_raw, psd_raw_microvolts, label='Raw PSD', alpha=0.7)  # Logarithmic scale
    plt.semilogy(f_clean, psd_clean_microvolts, label='Cleaned PSD', alpha=0.7)
    plt.axvspan(8, 12, color='green', alpha=0.2, label='Mu Band')
    plt.axvspan(13, 30, color='blue', alpha=0.2, label='Beta Band')
    plt.title(f'Power Spectral Density ({channel})')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power Spectral Density (µV²/Hz)')
    plt.legend()
    plt.show()



# Step 9: Compute SNR and Autocorrelation (before and after CCA)

# SIGNAL-TO-NOISE-RATIO (SNR)
from scipy.signal import welch
import numpy as np

# Extract data from MNE objects (raw data and cleaned data)
data_raw = matched_raw.get_data()  # Shape: (n_channels, n_samples)
data_clean = data_clean.transpose(1, 0, 2).reshape(data_raw.shape)  # Match raw shape if necessary

# Compute Welch PSD for each channel
fs = raw.info['sfreq']
f_raw, psd_raw = welch(data_raw, fs=fs, axis=-1, nperseg=1024)
f_clean, psd_clean = welch(data_clean, fs=fs, axis=-1, nperseg=1024)

# Compute SNR
def compute_snr(frequencies, psd, signal_band, noise_bands):
    # Compute signal power in the signal band
    signal_mask = (frequencies >= signal_band[0]) & (frequencies <= signal_band[1])
    signal_power = psd[signal_mask].sum()

    # Compute total noise power across all noise bands
    noise_power = 0
    for band in noise_bands:
        noise_mask = (frequencies >= band[0]) & (frequencies <= band[1])
        noise_power += psd[noise_mask].sum()

    # Return SNR
    return signal_power / noise_power

# Define signal and noise bands
signal_band = (8, 30)  # Mu and Beta bands
noise_bands = [(35, 45), (58, 62)]  # High-frequency, and powerline noise

# Compute SNR
snr_raw = compute_snr(f_raw, psd_raw.mean(axis=0), signal_band, noise_bands)
snr_clean = compute_snr(f_clean, psd_clean.mean(axis=0), signal_band, noise_bands)

print(f"SNR (Raw): {snr_raw:.2f}")
print(f"SNR (Cleaned): {snr_clean:.2f}")


# Select motor-related channels
motor_channels = ['C3', 'Cz', 'C4']
channel_indices = [raw.ch_names.index(ch) for ch in motor_channels]

# Extract and compute PSD for selected channels
f_raw_motor, psd_raw_motor = welch(data_raw[channel_indices], fs=fs, axis=-1, nperseg=1024)
f_clean_motor, psd_clean_motor = welch(data_clean[channel_indices], fs=fs, axis=-1, nperseg=1024)

# Average PSD for motor channels
psd_raw_motor_mean = psd_raw_motor.mean(axis=0)
psd_clean_motor_mean = psd_clean_motor.mean(axis=0)

# Compute SNR for motor channels
snr_raw_motor = compute_snr(f_raw_motor, psd_raw_motor_mean, signal_band, noise_bands)
snr_clean_motor = compute_snr(f_clean_motor, psd_clean_motor_mean, signal_band, noise_bands)

print(f"SNR (Raw, Motor Channels): {snr_raw_motor:.2f}")
print(f"SNR (Cleaned, Motor Channels): {snr_clean_motor:.2f}")



# Check if autocorrelation improves after CCA
# Reshape if necessary
if data_clean.ndim == 2:  # Flattened data
    n_epochs, n_channels, n_times = epochs.get_data().shape
    data_clean = data_clean.reshape(n_epochs, n_channels, n_times)
    print(f"Reshaped data_clean to: {data_clean.shape}")

# Get channel names and their indices
channel_names = epochs.info['ch_names']
motor_channels = ['C3', 'Cz', 'C4']
motor_indices = [channel_names.index(ch) for ch in motor_channels]
print(f"Indices of motor channels: {motor_indices}")

for idx, channel in zip(motor_indices, motor_channels):
    # Extract raw and cleaned signals for the current channel
    raw_signal = epochs.get_data()[:, idx, :].flatten()  # Flatten across epochs
    clean_signal = data_clean[:, idx, :].flatten()  # Flatten across epochs

    # Compute autocorrelation
    auto_raw = np.corrcoef(raw_signal[:-1], raw_signal[1:])[0, 1]
    auto_clean = np.corrcoef(clean_signal[:-1], clean_signal[1:])[0, 1]

    # Print results for the channel
    print(f"Channel: {channel}")
    print(f"Autocorrelation (Raw): {auto_raw:.2f}")
    print(f"Autocorrelation (Cleaned): {auto_clean:.2f}")




# Step 10: Save Cleaned Data
print("Saving cleaned data...")
cleaned_epochs = epochs.copy()
cleaned_epochs._data = data_clean
cleaned_epochs.save("cleaned_physionet_motor_imagery-epo.fif", overwrite=True)

print("Denoising complete! Cleaned data saved as 'cleaned_physionet_motor_imagery-epo.fif'.")
'''
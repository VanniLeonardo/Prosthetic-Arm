import mne 
import numpy as np
import matplotlib 
import matplotlib.pyplot as plt
from mne.filter import create_filter 
from sklearn.decomposition import PCA
import seaborn as sns 

# DISPLAY INFORMATION ABOUT THE DATASET

# set matplotlib backend
matplotlib.use('TkAgg')

sample_data_folder = mne.datasets.sample.data_path() # download the dataset "sample"
sample_data_raw_file = (
    sample_data_folder / "MEG" / "sample" / "sample_audvis_filt-0-40_raw.fif"
)
raw = mne.io.read_raw_fif(sample_data_raw_file) # displays some information like the number of porjection items (SSP prokjectors calculated to remove enviromental noise from MEG signals)

print(raw)
print(raw.info)

"""
OUTPUT: Opening raw data file C:\\Users\\Anna Notaro\\mne_data\\MNE-sample-data\\MEG\\sample\\sample_audvis_filt-0-40_raw.fif...
    Read a total of 4 projection items:  (used to remove noise from the data)
        PCA-v1 (1 x 102)  idle -> # number of components of the PCA vector
        PCA-v2 (1 x 102)  idle
        PCA-v3 (1 x 102)  idle
        Average EEG reference (1 x 60)  idle -> averaging the signals from all EEG electrodes to create a common reference
    Range : 6450 ... 48149 =     42.956 ...   320.665 secs # range of data points
Ready.
<Raw | sample_audvis_filt-0-40_raw.fif, 376  x 41700 (277.7 s), ~3.2 MB, data not loaded>
<Info | 14 non-empty values

channel description
 bads: 2 items (MEG 2443, EEG 053)
 ch_names: MEG 0113, MEG 0112, MEG 0111, MEG 0122, MEG 0123, MEG 0121, MEG ...
 chs: 204 Gradiometers, 102 Magnetometers, 9 Stimulus, 60 EEG, 1 EOG 
 custom_ref_applied: False
 dev_head_t: MEG device -> head transform
 dig: 146 items (3 Cardinal, 4 HPI, 61 EEG, 78 Extra)
 
 #highpass filter: highpass: 0.1 Hz
 hpi_meas: 1 item (list) Head Position Indicator (HPI) 
 hpi_results: 1 item (list)
 #lowpass: 40.0 Hz

 meas_date: 2002-12-03 19:01:10 UTC
 meas_id: 4 items (dict)
 nchan: 376
 projs: PCA-v1: off, PCA-v2: off, PCA-v3: off, Average EEG reference: off
 
 #sampling frequency: sfreq: 150.2 Hz 
 """

# PLOT THE RAW DATA AND COMPUTE THE POWER SPECTRAL DENSITY

raw.compute_psd(fmax=50).plot(picks="data", exclude="bads", amplitude=False)
raw.plot(duration=5, n_channels=30)
plt.show()

"""
POWER SPECTRAL DENSITY
measure how the power of a signal is distributed across different frequency components. 
power -> signal's mean square value (magnitude of the signal over time)
spectral -> shows how the signal's power is distributed across various frequencies
density -> normalization of the msv 

in how case the PSD plotshows the power distribution of the signal across different frequencies,

The PSD helps you identify which frequencies are present in your MEG/EEG data and how strong they are
By examining the PSD, you can identify and differentiate between signal and noise. Noise often appears as power at frequencies that are not typically associated with brain activity
The PSD can give you insights into the quality of your recorded signals. High power in certain frequency bands might indicate strong neural activity, while unexpected peaks might suggest artifacts or interference

X-Axis (Frequency): This axis represents the frequency in Hertz (Hz). It shows the range of frequencies present in your data.
Y-Axis (Power Density): This axis represents the power density, indicating how much power is present at each frequency.
Peaks: Peaks in the plot indicate frequencies where there is significant power. For example, a peak around 10 Hz might correspond to alpha waves, which are common in resting state EEG.
"""

# APPLY A BANDPASS FILTER TO THE DATA (ugly sorry)

sfreq = raw.info['sfreq']
l_freq = 10  
h_freq = 40

filtered_eeg_signal = mne.filter.filter_data(
    data = raw.get_data(),
    sfreq = sfreq,
    l_freq = l_freq,
    h_freq = h_freq,
    filter_length="auto",
    l_trans_bandwidth="auto",
    h_trans_bandwidth="auto",
    method="fir",
    iir_params=None,
    phase="zero",
    fir_window="hamming",
    fir_design="firwin",
    verbose=None,
    )
plt.figure() 
plt.plot(filtered_eeg_signal)
plt.title('Filtered EEG Signal')
plt.xlabel('Time (samples)')
plt.ylabel('Amplitude')
plt.xlim([300, 315])
plt.ylim([-50, 50])  # Set y-axis limits to decrease the interval represented
plt.tight_layout()
plt.show()

# ICA (Independent Component Analysis) to remove artifacts from the data

# set up and fit the ICA
ica = mne.preprocessing.ICA(n_components=20, random_state=97, max_iter=800)

"""
This initializes the ICA object with 20 components, a random state for reproducibility, 
and a maximum of 800 iterations for the algorithm to converge.
"""

ica.fit(raw)
"""
ica.fit(raw): This fits the ICA model to the raw EEG/MEG data. 
The model learns the unmixing matrix to separate the independent components from the mixed signals.
"""

ica.exclude = [1, 2]  # details on how we picked these are omitted here

"""
ica.exclude = [1, 2]: This specifies which components to exclude. 
These components are typically identified as artifacts (e.g., eye blinks, heartbeats).
"""

ica.plot_properties(raw, picks=ica.exclude)

"""
This plots the properties of the excluded components to visualize and confirm they are indeed artifacts.
"""

# Create a copy of the raw data before applying ICA
orig_raw = raw.copy()

# Ensure data is loaded into memory
raw.load_data()

# Apply ICA to remove artifacts
ica.apply(raw)

# Show some frontal channels to illustrate artifact removal
chs = [
    "MEG 0111", "MEG 0121", "MEG 0131", "MEG 0211", "MEG 0221", "MEG 0231",
    "MEG 0311", "MEG 0321", "MEG 0331", "MEG 1511", "MEG 1521", "MEG 1531",
    "EEG 001", "EEG 002", "EEG 003", "EEG 004", "EEG 005", "EEG 006",
    "EEG 007", "EEG 008",
]
chan_idxs = [raw.ch_names.index(ch) for ch in chs]

# Plot original raw data
orig_raw.plot(order=chan_idxs, start=12, duration=4)

# Plot cleaned raw data
raw.plot(order=chan_idxs, start=12, duration=4)

# Save the original raw data plot
orig_raw_fig = orig_raw.plot(order=chan_idxs, start=12, duration=4, show=False)
orig_raw_fig.savefig("original_raw_data.png")
# Save the cleaned raw data plot
cleaned_raw_fig = raw.plot(order=chan_idxs, start=12, duration=4, show=False)
cleaned_raw_fig.savefig("cleaned_raw_data.png")

# PCA (Principal Component Analysis) to reduce the dimensionality of the data

data = raw.get_data()
n_components = 10  # Adjust the number of components to your desired value
pca = PCA(n_components=n_components)
pca.fit(data.T)
# Transform the data
transformed_data = pca.transform(data.T)
info = mne.create_info(ch_names=[f'PCA-{i+1}' for i in range(n_components)], sfreq=raw.info['sfreq'])
transformed_raw = mne.io.RawArray(transformed_data.T, info)

transformed_raw.plot()

for i in range(n_components):
    for j in range(i + 1, n_components):
        plt.figure(figsize=(10, 7))
        sns.scatterplot(x=transformed_data[:, i], y=transformed_data[:, j])
        plt.title(f'Scatterplot of Principal Components {i+1} and {j+1}')
        plt.xlabel(f'Principal Component {i+1}')
        plt.ylabel(f'Principal Component {j+1}')
        plt.savefig(f'scatterplot_pca_{i+1}_vs_{j+1}.png')
        plt.close()




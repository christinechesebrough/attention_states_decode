#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 20 07:06:08 2025

@author: christinechesebrough
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
## Simplified script for loading epoched attention state data, extracting analytic power using hilbert transform, and saving
"""
import os, re, sys
from itertools import compress
import numpy as np
import pandas as pd
from scipy import stats, signal, interpolate
import matplotlib.pyplot as plt
import mne
import seaborn as sns
from mne.time_frequency import psd_array_welch

from fooof.plts.spectra import plot_spectrum


vid = 'betta' #'inscapes'
freq_band = 'theta'#'delta','theta','alpha','gamma','HFA'
ref = 'avg'
pow_type = 'log'

pat = 'NS217'

mne_data_dir = f'/Volumes/Samsung/Movie_data/movies_es_prep_standard/{pat}' #path to directory where the recordings are

# specific file path to data you want to process, you could also assign this dynamically
epoch_dat_path = '/Volumes/Samsung/Movie_data/movies_es_prep_standard/NS217/Neural_prep/NS217_ses-Exp_Samp_Inscapes01_behavior+ecephys_referenced_avg_probe_pre_onset-epoch.fif' #path to the file you want to process

# path to patient's correspondence file
excel_path = '/Volumes/Samsung/anatomy/NS217/elec_recon/NS217_Electrodes_Natus_TDT_correspondence_updated.xlsx'


out_dir = f'/Volumes/Samsung/Movie_data/movies_es_power/{pat}' # path to your output directory
if not os.path.exists(out_dir):
    os.makedirs(out_dir)


#%% Main script

# Define frequency band you want to extract

freq_band = 'alpha' 
if freq_band == 'alpha':
    freq_range = (8, 13)
elif freq_band == 'HFA':
    freq_range = (51, 150)
elif freq_band == 'delta':
    freq_range = (1, 3)
elif freq_band == 'theta':
    freq_range = (4, 7)
elif freq_band == 'beta':
    freq_range = (14, 30)
elif freq_band == 'gamma':
    freq_range = (31, 50)
elif freq_band == 'all_gamma':
    freq_range = (31,150)
elif freq_band == 'mid_gamma':
    freq_range = (50,69)
elif freq_band == 'theta_alpha':
    freq_range = (4,13)
elif freq_band == 'all':
    freq_range = (0,170)
else:
    raise Exception("no range assigned")
 
if freq_band == 'alpha':
    bin_width = 2
elif freq_band == 'beta':
    bin_width = 4
elif freq_band in ['gamma', 'all_gamma','mid_gamma']:
    bin_width = 8
elif freq_band == 'HFA':
    bin_width = 10
else:
    bin_width = 2  # default


# load epoched data
mne_data = mne.read_epochs(epoch_dat_path, preload=False)


# excel_path is path to this patient's correspondence sheet 
elecs_subs =pd.read_excel(excel_path)
required_cols = {"label"}
col_map = {col.lower(): col for col in elecs_subs.columns}
elecs_subs.rename(columns={col_map[req]: req for req in required_cols}, inplace=True)
      
# Visually inspect epoched data
plot_raw = True # can set to False if you don't want to visually inspect
if plot_raw:
    mne_data.plot(
        title=f"{pat} - {vid} - Probe-onset epochs: -12.1 to -0.1 s",
        scalings=dict(seeg=100e-6, ecog=100e-6),
        n_channels=64,
        show_scrollbars=True,
        n_epochs = 1,
        block=True
    )   

labels = mne_data.ch_names

ip_contacts = elecs_subs.label.values


## can filter included contacts by atlas label or other var from elec_subs 
#ip_contacts = elecs_subs.Contact.values[(elecs_subs.AparcAseg_Atlas != 'Right-Cerebral-White-Matter') & (elecs_subs.AparcAseg_Atlas != 'Left-Cerebral-White-Matter')]
ip_contacts = elecs_subs.loc[elecs_subs['Desikan_Killiany']!='Out','label'].values

if ref in ['wm_bip','bip']:
    labels_split = [l.split('-') for l in labels]
    idx_ip = np.unique(np.concatenate([np.where([np.sum([ld == ls for ls in lf]) for lf in labels_split])[0] for ld in ip_contacts]))
    idx_ip = np.in1d(np.arange(len(labels)), idx_ip)
elif ref in ['avg','wm']:
    idx_ip = np.array([label in ip_contacts for label in labels])

lfp = mne_data.get_data()
fs_lfp = mne_data.info['sfreq']
time_lfp = mne_data.times
    
lfp_ip = lfp[:,idx_ip, :]
    
labels_ip = list(compress(labels, idx_ip))

info = mne.create_info(ch_names=labels_ip,sfreq = fs_lfp)

f_start, f_end = freq_range
freq_bins = [(f, min(f + bin_width, f_end)) for f in range(f_start, f_end, bin_width)]

# Remove flat channels before processing (std < threshold)
flat_std_thresh = 1e-6

# Compute std across epochs and time for each channel
# result shape: (n_channels,)
stds = np.std(lfp_ip, axis=(0, 2))

nonflat_idx = stds > flat_std_thresh

# Keep non-flat channels
lfp_ip = lfp_ip[:, nonflat_idx, :]

labels_ip = [
    label for i, label in enumerate(labels_ip)
    if nonflat_idx[i]
]

print(f"After flat-channel removal: {lfp_ip.shape}")
print(f"Remaining labels: {len(labels_ip)}")


 #%%                   

# recreate MNE Info object after removing out and flat channels
info = mne.create_info(
    ch_names=labels_ip,
    sfreq=fs_lfp,
    ch_types=["ecog"] * len(labels_ip)
)

epochs_dat = mne.EpochsArray(
    lfp_ip,
    info,
    tmin=0.0,
    baseline=None
)

# psd_array_welch
psd, freqs = psd_array_welch(
    lfp_ip,
    sfreq=fs_lfp,
    fmin=freq_range[0],
    fmax=freq_range[1],
    n_fft=int(fs_lfp * 2),
    n_overlap=int(fs_lfp),
    average="mean"
)

print(f"PSD shape: {psd.shape}")  # n_epochs x n_channels x n_freqs

#%% #Extract power
    
raw_power_bins = []

for f_low, f_high in freq_bins:
    sos = signal.butter(
        5,
        [f_low, f_high],
        btype="bandpass",
        fs=fs_lfp,
        output="sos"
    )
    
    # Filter along time axis
    # input/output shape: n_epochs x n_channels x n_times
    band_bin = signal.sosfiltfilt(
        sos,
        lfp_ip,
        axis=2
    )
    
    # Hilbert along time axis
    analytic = signal.hilbert(
        band_bin,
        axis=2
    )
    
    # Envelope / amplitude
    power = np.abs(analytic)
    
    raw_power_bins.append(power)

# shape before mean: n_bins x n_epochs x n_channels x n_times
# shape after mean:  n_epochs x n_channels x n_times
pow_dat_raw = np.mean(np.stack(raw_power_bins, axis=0),axis=0)

if pow_type == "raw":
    pow_dat = pow_dat_raw
elif pow_type == "log":
    pow_dat = np.log10(pow_dat_raw + 1e-6)
else:
    raise ValueError(f"Unknown pow_type: {pow_type}")

t_lfp = np.arange(lfp_ip.shape[2]) / fs_lfp
   
# Create MNE Info object
info = mne.create_info(
    ch_names=labels_ip,
    sfreq=fs_lfp,
    ch_types=["ecog"] * len(labels_ip)
)

# If your epochs were originally -12.1 to -0.1 relative to probe onset
tmin_epoch = -12.1

pow_epochs = mne.EpochsArray(
    data=pow_dat,          # n_epochs x n_channels x n_times
    info=info,
    tmin=tmin_epoch,
    baseline=None,
    verbose=True
)

fig = pow_epochs.plot(
    title=f"{pat} {vid}- {ref.upper()} Analytic Amplitude Epochs",
    scalings=dict(ecog="auto", seeg="auto"),
    n_channels=64,
    n_epochs=5,
    events=True,
    show_scrollbars=True,
    block=True
)

#%%
# ------------------------------------------------------------
# Save pow_ip and labels
# ------------------------------------------------------------

out_fname = f"{pat}_{vid}_{ref}_{freq_band}_analytic_power_epochs.npz"
out_path = os.path.join(out_dir, out_fname)

np.savez(
    out_path,
    pow_epochs=pow_epochs,
    labels=np.asarray(labels_ip),
    fs_lfp=fs_lfp,
    t_lfp=t_lfp,
    freq_bins=np.asarray(freq_bins),
    pow_type=pow_type,
    pat=pat,
    vid=vid,
    ref=ref,
    )

print(f"Saved power data to: {out_path}")


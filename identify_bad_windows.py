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


vid = 'inscapes' #'inscapes'
ref = 'avg'
pow_type = 'log'

pat = 'LH019'

machine_path = '/media/christine'

mne_data_dir = f'{machine_path}/Samsung/Movie_data/exp_samp_prep_standard/{pat}/Neural_prep' #path to directory where the recordings are
# epoch_dat_path = os.path.join(
#     mne_data_dir,
#     f"{pat}_ses-Exp_Samp_Betta01_behavior+ecephys_referenced_avg_probe_pre_onset-epo.fif"
# )

#epoch_dat_path =
#'/media/christine/Samsung/Movie_data/exp_samp_prep_standard/LH019/Neural_prep/LH019_ses-Exp_Samp_Betta_combined_behavior+ecephys_referenced_avg_probe_pre_onset-epo.fif'
epoch_dat_path = f'/media/christine/Samsung/Movie_data/exp_samp_prep_standard/{pat}/Neural_prep/{pat}_ses-Exp_Samp_Inscapes01_behavior+ecephys_referenced_avg_probe_pre_onset-epoch.fif'

# path to patient's correspondence file
excel_path = f'{machine_path}/Data/anatomy/LH019/elec_recon/LH019_Electrodes_Natus_TDT_correspondence_updated.xlsx'


out_dir = f'{machine_path}/Samsung/Movie_data/movies_es_power/{pat}' # path to your output directory
if not os.path.exists(out_dir):
    os.makedirs(out_dir)


# Load original 12-second epochs
mne_data = mne.read_epochs(
    epoch_dat_path,
    preload=True
)

# excel_path is path to this patient's correspondence sheet 
elecs_subs =pd.read_excel(excel_path)
required_cols = {"label"}
col_map = {col.lower(): col for col in elecs_subs.columns}
elecs_subs.rename(columns={col_map[req]: req for req in required_cols}, inplace=True)

labels = mne_data.ch_names
ip_contacts = elecs_subs.label.values


ip_contacts = elecs_subs.loc[elecs_subs['Desikan_Killiany']!='Out','label'].values


# Keep the original object unchanged
epochs_for_qc = mne_data.copy()


# Optionally retain only included anatomical contacts
labels_ip = [
    label
    for label in epochs_for_qc.ch_names
    if label in ip_contacts
]

epochs_for_qc.pick(labels_ip)
      
# # Visually inspect epoched data
# plot_raw = True # can set to False if you don't want to visually inspect
# if plot_raw:
#     mne_data.plot(
#         title=f"{pat} - {vid} - Probe-onset epochs: -12.1 to -0.1 s",
#         scalings=dict(seeg=100e-6, ecog=100e-6),
#         n_channels=64,
#         show_scrollbars=True,
#         n_epochs = 1,
#         block=True
#     )   


window_duration = 1.0
sfreq = epochs_for_qc.info["sfreq"]
samples_per_window = int(round(window_duration * sfreq))

data = epochs_for_qc.get_data()
n_parent_epochs, n_channels, n_times = data.shape

n_windows = n_times // samples_per_window
n_samples_used = n_windows * samples_per_window

print(f"Original epochs: {n_parent_epochs}")
print(f"One-second windows per epoch: {n_windows}")
print(f"Unused endpoint samples: {n_times - n_samples_used}")

# Discard only a possible extra endpoint sample
data = data[:, :, :n_samples_used]

# parent epoch × channel × window × sample
window_data = data.reshape(
    n_parent_epochs,
    n_channels,
    n_windows,
    samples_per_window
)

# parent epoch × window × channel × sample
window_data = window_data.transpose(0, 2, 1, 3)

# Combine parent epoch and window into the Epochs dimension
window_data = window_data.reshape(
    n_parent_epochs * n_windows,
    n_channels,
    samples_per_window
)


window_numbers = np.tile(
    np.arange(n_windows),
    n_parent_epochs
)

parent_epochs = np.repeat(
    np.arange(n_parent_epochs),
    n_windows
)

window_start_times = (
    epochs_for_qc.tmin
    + window_numbers * window_duration
)

code_to_label = {
    code: label
    for label, code in epochs_for_qc.event_id.items()
}

parent_event_codes = np.repeat(
    epochs_for_qc.events[:, 2],
    n_windows
)

window_metadata = pd.DataFrame({
    "window_id": np.arange(len(window_data)),
    "parent_epoch": parent_epochs,
    "original_selection": np.repeat(
        epochs_for_qc.selection,
        n_windows
    ),
    "parent_event_code": parent_event_codes,
    "parent_event_label": [
        code_to_label.get(code, str(code))
        for code in parent_event_codes
    ],
    "window": window_numbers,
    "window_start": window_start_times,
    "window_end": window_start_times + window_duration
})

if epochs_for_qc.metadata is not None:
    repeated_metadata = (
        epochs_for_qc.metadata
        .iloc[parent_epochs]
        .reset_index(drop=True)
    )

    window_metadata = pd.concat(
        [
            window_metadata.reset_index(drop=True),
            repeated_metadata
        ],
        axis=1
    )
    
window_events = np.column_stack([
    np.arange(len(window_data)) * samples_per_window,
    np.zeros(len(window_data), dtype=int),
    np.ones(len(window_data), dtype=int)
])

qc_epochs = mne.EpochsArray(
    window_data,
    epochs_for_qc.info.copy(),
    events=window_events,
    event_id={"qc_window": 1},
    tmin=0.0,
    metadata=window_metadata,
    baseline=None
)

# Preserve a complete catalog before rejected windows are dropped
qc_table = qc_epochs.metadata.copy()

qc_epochs.plot(
    title=f"{pat} - {vid} - 1-second QC windows",
    scalings=dict(seeg=150e-6, ecog=100e-6),
    n_channels=min(64, len(qc_epochs.ch_names)),
    n_epochs=10,  # displays one original trial at a time
    show_scrollbars=True,
    block=True
)

kept_window_ids = set(
    qc_epochs.metadata["window_id"].astype(int)
)

qc_table["bad"] = ~qc_table["window_id"].isin(
    kept_window_ids
)

qc_table["rejection_reason"] = np.where(
    qc_table["bad"],
    "USER",
    ""
)

qc_csv_path = os.path.join(
    out_dir,
    f"{pat}_{vid}_{ref}_1s_window_qc.csv"
)

qc_table.to_csv(
    qc_csv_path,
    index=False
)

clean_epochs_path = os.path.join(
    out_dir,
    f"{pat}_{vid}_{ref}_clean_1s-epo.fif"
)

if os.path.exists(clean_epochs_path):
    response = input(f"File already exists:\n{clean_epochs_path}\nOverwrite? [y/N]: ").strip().lower()

    if response in ["y", "yes"]:
        qc_epochs.save(clean_epochs_path, overwrite=True)
    else:
        print("Save cancelled.")
else:
    qc_epochs.save(clean_epochs_path, overwrite=False)

qc_epochs.save(
    clean_epochs_path,
    overwrite=True
)

print(f"Rejected windows: {qc_table['bad'].sum()}")
print(f"Retained windows:  {(~qc_table['bad']).sum()}")
print(f"QC table saved to: {qc_csv_path}")
print(f"Clean epochs saved to: {clean_epochs_path}")
#%%
window_duration = 1.0
sfreq = mne_data.info["sfreq"]
samples_per_window = int(round(window_duration * sfreq))

data = mne_data.get_data()
n_epochs, n_channels, n_times = data.shape

n_windows = n_times // samples_per_window
n_samples_used = n_windows * samples_per_window

# Remove any extra endpoint sample before reshaping
data = data[:, :, :n_samples_used]

subepoch_data = (
    data.reshape(
        n_epochs,
        n_channels,
        n_windows,
        samples_per_window
    )
    .transpose(0, 2, 1, 3)
    .reshape(
        n_epochs * n_windows,
        n_channels,
        samples_per_window
    )
)

#%%


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


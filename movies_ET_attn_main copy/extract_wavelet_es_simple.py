#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 25 13:20:46 2026

@author: christinechesebrough
"""

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
from mne.time_frequency import tfr_array_morlet, tfr_array_multitaper

vid = 'betta' #'inscapes'
freq_band = 'theta'#'delta','theta','alpha','gamma','HFA'
ref = 'avg'

tf_freq_range = (2, 150)   
freq_step = 2             
decim_tf = 2               

output = 'morlet'
n_cycles_mode = 'bounded'   # 'scaled' 'bounded' or 'fixed'
fixed_n_cycles = 5

save_continuous_tf = False

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

# load epoched data
mne_data = mne.read_epochs(epoch_dat_path, preload=False)


# excel_path is path to this patient's correspondence sheet 
elecs_subs =pd.read_excel(excel_path)
required_cols = {"label"}
col_map = {col.lower(): col for col in elecs_subs.columns}
elecs_subs.rename(columns={col_map[req]: req for req in required_cols}, inplace=True)
      
# Visually inspect epoched data
plot_raw = False # can set to False if you don't want to visually inspect
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


f_start, f_end = tf_freq_range
freqs_tf = np.arange(f_start, f_end + freq_step, freq_step)
                
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


#%% #Extract wavelet
    
if output == 'morlet':

    # lfp_ip is epoched, so shape = (n_epochs, n_channels, n_times)
    data_tf = lfp_ip

    if n_cycles_mode == 'scaled':
        n_cycles = freqs_tf / 2.0
    
    elif n_cycles_mode == 'bounded':
        n_cycles = np.clip(freqs_tf / 2.0, 3, 12)
    
    elif n_cycles_mode == 'fixed':
        n_cycles = fixed_n_cycles
    
    
    power_tf = tfr_array_morlet(
        data_tf,
        sfreq=fs_lfp,
        freqs=freqs_tf,
        n_cycles=n_cycles,
        output='power',
        decim=decim_tf,
        n_jobs=1
    )

    # power_tf shape: (n_epochs, n_channels, n_freqs, n_times_decim)

    if pow_type == 'log':
        power_tf = np.log10(power_tf + 1e-6)

    pow_tf_dat = power_tf

    # Preserve epoch-relative time, e.g. -12.1 to -0.1 s
    t_tf = mne_data.times[::decim_tf]

    fs_tf = fs_lfp / decim_tf

    print(f"Wavelet power shape: {pow_tf_dat.shape}")
    print(f"Time vector shape: {t_tf.shape}")


elif output == 'multitaper':

    # lfp_ip is already epoched, so shape = (n_epochs, n_channels, n_times)
    data_tf = lfp_ip

    if n_cycles_mode == 'scaled':
        n_cycles = freqs_tf / 2.0
    
    elif n_cycles_mode == 'bounded':
        n_cycles = np.clip(freqs_tf / 2.0, 3, 12)
    
    elif n_cycles_mode == 'fixed':
        n_cycles = fixed_n_cycles

    power_tf = tfr_array_multitaper(
        data_tf,
        sfreq=fs_lfp,
        freqs=freqs_tf,
        n_cycles=n_cycles,
        output='power',
        decim=decim_tf,
        time_bandwidth=4.0,
        n_jobs=1
    )

    # power_tf shape: (n_epochs, n_channels, n_freqs, n_times_decim)

    if pow_type == 'log':
        power_tf = np.log10(power_tf + 1e-6)

    pow_tf_dat = power_tf

    # Preserve epoch-relative time
    t_tf = mne_data.times[::decim_tf]

    fs_tf = fs_lfp / decim_tf

    print(f"Multitaper power shape: {pow_tf_dat.shape}")
    print(f"Time vector shape: {t_tf.shape}")
    
         
#
#%% Save TF output as .npz

out_fname = f"{pat}_{vid}_{ref}_{output}_{tf_freq_range[0]}-{tf_freq_range[1]}Hz_wavelet_epochs.npz"
out_path = os.path.join(out_dir, out_fname)

np.savez_compressed(
    out_path,
    pow_tf_dat=pow_tf_dat,              # shape: n_epochs x n_channels x n_freqs x n_times
    freqs_tf=freqs_tf,                  # frequency vector
    t_tf=t_tf,                          # epoch-relative time vector after decimation
    fs_tf=fs_tf,                        # effective sampling rate after decimation
    labels_ip=np.array(labels_ip),      # included channel labels
    pat=pat,
    vid=vid,
    ref=ref,
    output=output,
    pow_type=pow_type,
    tf_freq_range=np.array(tf_freq_range),
    freq_step=freq_step,
    decim_tf=decim_tf,
    n_cycles=n_cycles,
    n_cycles_mode=n_cycles_mode,
    fs_lfp=fs_lfp,
)

print(f"Saved TF data to: {out_path}")
print(f"Saved power shape: {pow_tf_dat.shape}")




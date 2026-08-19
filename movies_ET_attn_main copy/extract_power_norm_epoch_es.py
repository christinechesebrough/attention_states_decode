#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 20 07:06:08 2025

@author: christinechesebrough
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Updated December 16th
Only includes neural data (not eye features, computed elsewhere)
Added finding of peak freqs using FOOOF

Correlate eye movement based ISC measures with neural signals suspected to
index attentional state changes 
"""
import os, re, sys
from scipy.stats import pearsonr
from itertools import compress
import numpy as np
import pandas as pd
from scipy import stats, signal, interpolate
import matplotlib.pyplot as plt
import mne
import seaborn as sns
from mne.time_frequency import psd_array_welch


machine_path ='media/christine'#'Volumes' #'media/christine'


# Add Linux library paths BEFORE importing epipe
sys.path.insert(0, f'/{machine_path}/Samsung/EPIPE/Python')
sys.path.insert(0, f'/{machine_path}/Samsung/iEEG2NWB-main')

vids = ['inscapes']# 'betta'
freq_bands = ['theta','alpha','beta','gamma','HFA']#,'alpha','beta','gamma','HFA']#['delta','theta','alpha','gamma','HFA']#'beta','gamma','HFA'] #'delta','theta','alpha','beta','gamma'

ref = 'avg'
region = 'all'

data_dir =  f'/{machine_path}/Samsung/Movie_data/exp_samp_prep_standard'
mne_data_dir =  f'/{machine_path}/Samsung/Movie_data/exp_samp_prep_standard'
elec_dir = f'/{machine_path}/Samsung/Movie_data/data/electrode_localization'
fs_dir = f'/{machine_path}/Samsung/anatomy'
corr_dir = f'/{machine_path}/Samsung/Movie_data/data/movie_elec_corr_sheets'

fs_eye = 300

plot_power = False
plot_power_subsets = True
extract_power = True

output = 'power_z'
pow_type = 'log'

wd = '/Volumes/Samsung/scripts/movies_ET_attn_main'
src_dir = os.path.join(wd, 'src')
src_dir = os.path.abspath(src_dir)

# Add src to path if not already there
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

# 1) Put src on sys.path (at the front)
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)
print('on sys.path?', src_dir in sys.path)

#%%

def _extract_run_label(fname: str) -> str:
    """
    Try to extract a run label like 'run-01' or 'run-1' from a filename.
    Fallback: 'run-01'.
    """
    m = re.search(r'run[-_]?(\d+)', fname, flags=re.IGNORECASE)
    if m:
        return f"run-{int(m.group(1)):02d}"
    return "run-01"

def _extract_ses_label(fname: str) -> str:
    """
    Optional: extract 'ses-02' etc. Fallback: ''.
    """
    m = re.search(r'ses[-_]?(\d+)', fname, flags=re.IGNORECASE)
    if m:
        return f"ses-{int(m.group(1)):02d}"
    return ""

def _sort_key(f):
    ses = _extract_ses_label(f)
    run = _extract_run_label(f)
    return (ses, run, f)


#%% Main script

vids.sort()
freq_bands.sort()

processed_lfp_files = []  # initialize once


for vid in vids:
    
    if vid == "Betta":
        patients = ['LH019','NS219','NS221','NS217']
       
   
    elif vid == 'inscapes':
        patients = ['LH019','NS219','NS217'] 
   
    patients.sort()
        
    if vid == 'inscapes':
        keys = ['inscapes']
    if vid == 'Betta':
        keys = ['Betta','betta']    
            
    #%
    for pat in patients:
        pat_dir = os.path.join(data_dir, pat)
            
        lfp_pat_dir = '{:s}/{:s}/Neural_prep'.format(mne_data_dir, pat)
        
        lfp_files = os.listdir(lfp_pat_dir)
                    
        lfp_files = [
            f for f in os.listdir(lfp_pat_dir)
            if f.endswith(".fif")
            and ref in f
            and ('referenced' in f)
            and ('probe_annotations' in f)
            and 'aic' not in f
            and any(k.lower() in f.lower() for k in keys)
            and not f.startswith("._")
        ]
        
        # Deterministic sort: by session then run then filename

        lfp_files = sorted(lfp_files, key=_sort_key)

        print(f"Found {len(lfp_files)} matching runs for {pat}:")
        for f in lfp_files:
            print("  ", f)
            
            processed_lfp_files.append(f)

        # # Iterate over each file/run as an independent entry
        for lfp_file in lfp_files:
            # Build an entry label that will propagate to outputs
            ses_label = _extract_ses_label(lfp_file)
            run_label = _extract_run_label(lfp_file)
            if ses_label:
                entry_id = f"{pat}_{ses_label}_{run_label}"
            else:
                entry_id = f"{pat}_{run_label}"

            print(f"Loading data for entry {entry_id} from {lfp_file} ...")
            
            if run_label == 'run-01':
                run_keys = ['run-01','run-1']
            if run_label == 'run-02':
                run_keys = ['run-02','run-2']

            mne_data = mne.io.read_raw(os.path.join(lfp_pat_dir, lfp_file), preload=False)
  
            sub_fs_dir = '{:s}/{:s}'.format(fs_dir, pat)
              
            subid = os.path.basename(sub_fs_dir)

            elec_recon_dir = corr_dir    

            excel_files = sorted([
                f for f in os.listdir(elec_recon_dir)
                if pat in f
                and f.endswith('.xlsx')
                and not f.startswith('.')
            ])
            if not excel_files:
                print(f"  [SKIP] No correspondence .xlsx for {pat} found in {elec_recon_dir}")
                continue

            # Use most recently modified if multiple exist (matches Python HFO script)
            excel_path = max(
                [os.path.join(elec_recon_dir, f) for f in excel_files],
                key=os.path.getmtime
            )
            
            elecs_subs =pd.read_excel(excel_path)
            required_cols = {"label"}
            col_map = {col.lower(): col for col in elecs_subs.columns}
            elecs_subs.rename(columns={col_map[req]: req for req in required_cols}, inplace=True)

            
            # Find candidate files already in the patient dir
            bad_channel_files = [
                f for f in os.listdir(lfp_pat_dir)
                if (
                    ('bad_channels' in f)
                    and f.endswith('.txt')
                    and any(k.lower() in f.lower() for k in keys)
                    and not f.startswith('._')
                )
            ]
            
            # Narrow to candidates matching ref + run
            ref_bad_channel_files = [
                f for f in bad_channel_files
                if (
                    (ref in f)
                    and any(r in f for r in run_keys)
                )
            ]
            
            # Helper: pick most recent if multiple
            def pick_most_recent(files):
                if not files:
                    return None
                return max(files, key=lambda fn: os.path.getmtime(os.path.join(lfp_pat_dir, fn)))
            
            picked = pick_most_recent(ref_bad_channel_files)
            
            # If none found, define a NEW file name we will create
            # Make sure this naming is unique enough for your workflow
            if picked is None:
                # You can include ses_label/run_label if you want; run_keys might be list like ["run-01", "run-1"]
                # Use run_label if you have a single normalized run label available.
                picked = f'{pat}_{ses_label}_{vid}_{run_label}_{ref}_bad_channels.txt'

            bad_channel_path = os.path.join(lfp_pat_dir, picked)
            
            # Read existing file (if any)
            #    Preserve header/comments, parse channel lines robustly.
                           # Read existing file (if any)
            header_lines = []
            existing_bads = []
            
            if os.path.exists(bad_channel_path):
                with open(bad_channel_path, "r") as f:
                    for ln in f:
                        s = ln.strip()
                        if not s:
                            continue
                        if s.startswith("#"):
                            header_lines.append(ln.rstrip("\n"))
                            continue
                        existing_bads.append(s)
            
            # Safely get FIF bads (empty list if none)
            fif_bads = list(mne_data.info.get("bads", []))
            
            # --- NEW: filter to only channels that exist in this Raw ---
            ch_set = set(mne_data.ch_names)  # or mne_data.info["ch_names"]
            
            existing_bads_valid = [ch for ch in existing_bads if ch in ch_set]
            existing_bads_missing = [ch for ch in existing_bads if ch not in ch_set]
            
            # Optional: log missing bads once (useful for debugging)
            if existing_bads_missing:
                print(
                    f"Warning: {len(existing_bads_missing)} bad channels from file are not in this recording and will be ignored. "
                    f"Examples: {existing_bads_missing[:10]}"
                )
            
            # Merge (dedupe, preserve order) using only valid names
            merged_bads = list(dict.fromkeys(fif_bads + existing_bads_valid))
            
            # Assign back
            mne_data.info["bads"] = merged_bads

            # Interactive marking

            if visualize_mne_steps:
                mne_data.plot(
                    scalings=dict(seeg=200e-6),
                    n_channels=32,
                    remove_dc=True,
                    show_scrollbars=True,
                    duration=12.0,
                    block=True
                )
            
            # Save bads after closing plot
            final_bads = list(dict.fromkeys(mne_data.info.get("bads", [])))  # dedupe again
            
            os.makedirs(lfp_pat_dir, exist_ok=True)
            
            tmp_path = bad_channel_path + ".tmp"
            with open(tmp_path, "w") as f:
                # If file had no header, write a minimal one
                if not header_lines:
                    f.write(f"# Bad channels for {pat} {vid} {run_label} {ref}\n")
                else:
                    f.write("\n".join(header_lines) + "\n")
            
                # Write channel names, one per line
                if final_bads:
                    f.write("\n".join(final_bads) + "\n")
            
            os.replace(tmp_path, bad_channel_path)
            print(f"Bad channels saved to: {bad_channel_path}")
            
            # Drop them for downstream processing

            if final_bads:
                mne_data.drop_channels(final_bads)
            
                            
            labels = mne_data.ch_names
                                    
            exclude_strings = ['bankssts']
        
            #ip_contacts = elecs_subs.Contact.values
            ip_contacts = elecs_subs.label.values

           # ip_contacts = elecs_subs.Contact.values[(elecs_subs.AparcAseg_Atlas != 'Right-Cerebral-White-Matter') & (elecs_subs.AparcAseg_Atlas != 'Left-Cerebral-White-Matter')]
            # ip_contacts = elecs_subs.loc[elecs_subs['DK_Lobe'].isin(['Right-Hippocampus', 'Left-Hippocampus']),
            #     'Contact'
            # ].values
            if len(ip_contacts) == 0:
                pass
            else:
                if ref == 'wm_bip':
                    labels_split = [l.split('-') for l in labels]
                    idx_ip = np.unique(np.concatenate([np.where([np.sum([ld == ls for ls in lf]) for lf in labels_split])[0] for ld in ip_contacts]))
                    idx_ip = np.in1d(np.arange(len(labels)), idx_ip)
                if ref in ['avg','wm']:
                    idx_ip = np.array([label in ip_contacts for label in labels])
        
               # lfp = mne_data.get_data()
                fs_lfp = mne_data.info['sfreq']
                time_lfp = mne_data.times
                                        
                labels_ip = list(compress(labels, idx_ip))
                lfp_ip = mne_data.get_data(picks=labels_ip)

                info = mne.create_info(ch_names=labels_ip,sfreq = fs_lfp)
  
    
                # Remove flat channels before processing (std < threshold)
                flat_std_thresh = 1e-6
                stds = np.std(lfp_ip, axis=1)
                nonflat_idx = stds > flat_std_thresh
                lfp_ip = lfp_ip[nonflat_idx]
                labels_ip = [label for i, label in enumerate(labels_ip) if nonflat_idx[i]]

                fooof_region = None
                
                freq_band_count = 0
                
                for freq_band in freq_bands:
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
            
                    print(f"Processing data for {vid} in {freq_band} with range: {freq_range} Hz")
                    if output == 'entropy':
                        fig_dir = f'/{machine_path}/Samsung/Movie_data/{freq_band}_{region}_{vid}_all_cortContacts_entropy_29Mar25/entropy_extracted'
                    else:
                        fig_dir = f'/{machine_path}/Samsung/Movie_data/{output}_{pow_type}_{freq_band}_{vid}_26Mar26'
                    if not os.path.exists(fig_dir):
                        os.makedirs(fig_dir)
                    
                    fig_patient_dir = os.path.join(fig_dir, pat)
                    if not os.path.exists(fig_patient_dir):
                        os.makedirs(fig_patient_dir)
                    
                    f_start, f_end = freq_range
                    freq_bins = [(f, min(f + bin_width, f_end)) for f in range(f_start, f_end, bin_width)]
                                
              
#%%
                    if plot_power:
                        # Create an MNE Info object (still useful for later if you want)
                        info = mne.create_info(ch_names=labels_ip, sfreq=fs_lfp,
                                               ch_types=['ecog'] * len(labels_ip))
                        lfp_dat = mne.io.RawArray(lfp_ip, info)
                    
                        # Compute the PSD directly from the NumPy array
                        psd, freqs = psd_array_welch(
                            lfp_ip,                    # <--- use array, not RawArray
                            sfreq=fs_lfp,
                            fmin=freq_range[0],
                            fmax=freq_range[1],
                            n_fft=int(fs_lfp * 2),
                            n_overlap=int(fs_lfp),
                            average='mean'
                        )
                        
                        
                    if output == 'power_z':
                        pow_dat = np.zeros(
                            lfp_ip.shape,
                            dtype=np.float64
                        )
                        
                        n_bins_used = 0
                        
                        for f_low, f_high in freq_bins:
                        
                            print(f"    Processing {f_low}-{f_high} Hz")
                        
                            sos = signal.butter(5,[f_low, f_high],btype='bandpass',fs=fs_lfp, output='sos')
                        
                            band_bin = signal.sosfiltfilt(sos,lfp_ip,axis=1)
                        
                            analytic = signal.hilbert(band_bin,axis=1)
                        
                            power = np.abs(analytic)
                        
                            # No longer need these large arrays
                            del band_bin
                            del analytic
                        
                            if pow_type == 'raw':
                                power_for_z = power
                        
                            elif pow_type == 'log':
                                power_for_z = np.log10(power + 1e-6)
                                del power
                        
                            else:
                                raise ValueError(
                                    f"Unknown pow_type: {pow_type}"
                                )
                        
                            median_power = np.median(power_for_z,axis=1,keepdims=True)
                        
                            mad_power = np.median(np.abs(power_for_z - median_power),axis=1,keepdims=True)
                        
                            robust_std = 1.4826 * mad_power
                            robust_std[robust_std == 0] = np.nan
                        
                            z_power = (power_for_z - median_power) / robust_std
                        
                            # Add this bin directly to accumulator
                            np.nan_to_num(z_power, copy=False)
                            pow_dat += z_power
                            
                            n_bins_used += 1
                        
                            del power_for_z
                            del z_power
                            del median_power
                            del mad_power
                            del robust_std
                        
                        pow_dat /= n_bins_used
                        
                    elif output == 'power':
                        raw_power_bins = []
    
                        for f_low, f_high in freq_bins:
                            sos = signal.butter(5, [f_low, f_high], btype='bandpass', fs=fs_lfp, output='sos')
                            band_bin = signal.sosfiltfilt(sos, lfp_ip, axis=1)
                            
                            analytic = signal.hilbert(band_bin, axis=1)
                            power = np.abs(analytic)  # raw envelope
                            raw_power_bins.append(power)
                    
                        pow_dat_raw = np.mean(np.stack(raw_power_bins, axis=0), axis=0)  # (n_channels, n_times)
                    
                        if pow_type == 'raw':
                            pow_dat = pow_dat_raw
                        elif pow_type == 'log':
                            pow_dat = np.log10(pow_dat_raw + 1e-6)
                        else:
                            raise ValueError(f"Unknown pow_type: {pow_type}")
                            
                    else:
                        raise ValueError(f"Unknown output: {output}")
                                                
                    t_lfp = np.arange(lfp_ip.shape[1]) / fs_lfp
    

                
                    if plot_power:
                       time_isc = np.arange(1, pow_dat.shape[1] + 1)
                   
                       # Downsample only for plotting if not using rolling average
                       if not rolling_average:
                           ds_factor = 2400  # adjust (e.g., 5, 10, 20 depending on density)
                           time_isc_plot = time_isc[::ds_factor]
                           pow_dat_plot = pow_dat[:, ::ds_factor]
                       else:
                           time_isc_plot = time_isc
                           pow_dat_plot = pow_dat
                   
                       fig, ax = plt.subplots(figsize=(10, 5))
                   
                       for i, channel in enumerate(labels_ip):
                           ax.plot(time_isc_plot, pow_dat_plot[i, :], label=channel)
                   
                       ax.set_title('Interpolated Power Data')
                       ax.set_xlabel('Window')
                       ax.set_ylabel('Power')
                       ax.grid(True)
                   
                       plt.show()
                   #%% Epoch 
                       
                  # Find QSTART annotations
                    qstart_mask = np.array([
                        desc.startswith("QSTART")
                        for desc in mne_data.annotations.description
                    ])
                    
                    qstart_onsets_s = mne_data.annotations.onset[qstart_mask]
                    qstart_labels = mne_data.annotations.description[qstart_mask]
                    
                    
                    # Convert annotation times to sample indices
                    qstart_samples = mne_data.time_as_index(qstart_onsets_s)
                    
                    # Epoch parameters
                    epoch_tmin = -12.1
                    epoch_tmax = -.1
                    
                    epoch_start_offset = int(epoch_tmin * fs_lfp)
                    epoch_end_offset = int(epoch_tmax * fs_lfp)
                    
                    expected_n_samples = epoch_end_offset - epoch_start_offset

                    power_epochs = []
                    epoch_labels = []
                    

                    for sample, label in zip(qstart_samples, qstart_labels):
                    
                        start_sample = sample + epoch_start_offset
                        end_sample = sample + epoch_end_offset
                    
                        # # Must fall fully within the recording
                        if start_sample < 0 or end_sample > pow_dat.shape[1]:
                            print(f"Skipping {label}: epoch extends outside recording")
                            continue
                    
                        epoch = pow_dat[:, start_sample:end_sample]
                    
                        # # Only keep complete 12-second epochs
                        if epoch.shape[1] != expected_n_samples:
                            print(
                                f"Skipping {label}: expected {expected_n_samples} samples, "
                                f"got {epoch.shape[1]}"
                            )
                            continue
                    
                        power_epochs.append(epoch)
                        epoch_labels.append(label)
                    
                    power_epochs = np.stack(power_epochs, axis=0)
                    
                    if pat == 'LH019':
                        epoch_labels = np.array([f"QSTART_{i+1}" for i in range(len(epoch_labels))])
                        if power_epochs.shape[0] == 16:
                              power_epochs = power_epochs[:-3]
                              epoch_labels = epoch_labels[:-3]
                    print(power_epochs.shape)

    #%% save normed power epoched data
                    pow_epochs_output = os.path.join(fig_patient_dir,f"{entry_id}_{output}_{pow_type}_epoch_dat.npz")
                    
                    np.savez(
                        pow_epochs_output,
                        power_epochs=power_epochs,
                        epoch_labels=np.array(epoch_labels),
                        channel_names=np.array(labels_ip),
                        sfreq=fs_lfp,
                        tmin=epoch_tmin,
                        tmax=epoch_tmax,
                        freq_bins=np.array(freq_bins),
                        output=output,
                        pow_type=pow_type
                    )
              
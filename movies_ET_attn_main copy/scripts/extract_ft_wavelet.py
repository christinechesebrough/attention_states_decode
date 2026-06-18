#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 11:15:48 2026

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
from mne.time_frequency import tfr_array_morlet, tfr_array_multitaper
import multiprocessing

from scipy.signal import decimate
from scipy.signal import correlate

from joblib import Parallel, delayed


from fooof import FOOOF
from fooof.plts.spectra import plot_spectrum
#from antropy import sample_entropy, spectral_entropy, perm_entropy, lziv_complexity

# Add Linux library paths BEFORE importing epipe
# sys.path.insert(0, '/media/christine/Samsung/EPIPE/Python')
# sys.path.insert(0, '/media/christine/Samsung/iEEG2NWB-main')


sys.path.insert(0, '/Volumes/Samsung/EPIPE-movie_nwb/Python')
sys.path.insert(0, '/Volumes/Samsung/iEEG2NWB-main')


#vids = ['inscapes','despicable_me_english']#,'despicable_me_english']
vids = ['despicable_me_english','inscapes']#,'despicable_me_english']#,'despicable_me_english']

drive = 'Samsung'
ref = 'avg'
#freq_band = 'HFA'
region = 'all'

machine_path = 'Volumes'#'Volumes' #'media/christine'


data_dir = f'/{machine_path}/Samsung/Movie_data/movies_prep_standard'
isc_dir = f'/{machine_path}/SamsungMovie_data/data/isc'
mne_data_dir = f'/{machine_path}/Samsung/Movie_data/movies_prep_standard'
elec_dir = f'/{machine_path}/Samsung/Movie_data/data/electrode_localization'
fs_dir = f'/{machine_path}/Samsung/anatomy'

corr_dir = f'/{machine_path}/Samsung/Movie_data/data/movie_elec_corr_sheets'

fs_eye = 300

visualize_mne_steps = False
condense_to_isc = False
rolling_average = False
lowpass = False
find_peaks = False
plot_power = True
plot_power_subsets = False
use_interpolation = False
window_compare = False

output = 'wavelet'
pow_type = 'log'

save_continuous_tf = True


tf_freq_range = (3, 7)   # broad spectrum to preserve
freq_step = 1              # or 1 if you want denser sampling
decim_tf = 2               # 2 is fine, 4 may be more practical
n_cycles_mode = 'fixed'   # 'scaled' or 'fixed'
fixed_n_cycles = 5

save_continuous_tf = False

window_length_sec = 10
overlap_sec = 7.5
target_num_steps = 236


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


from epipe import nwb2mne, inspectNwb
from epipe import inspectNwb, nwb2mne, read_ielvis
import sys

sys.path.insert(0, '/media/christine/Samsung/scripts/movies_ET_attn_main/src')
# Import EEG preprocessing helper functions
from eeg_preproc_helpers import (
    plot_power_spectra,  plot_psd_batched, 
    create_file_paths, apply_highpass_filter,
    save_bad_channels, load_bad_channels, check_bad_channels_integrity, load_preprocessed_data, validate_mne_structure, preserve_mne_structure, restore_mne_structure, load_data_with_bad_channels, summarize_bad_channels, 
    interpolate_spikes, make_groups_from_prefix, regress_out_noise_by_group,
    detect_spikes_all_channels, reref_avg_by_group,
    ProcessingLogger, detect_spikes_ref1)

###### EXTRACT NORMALIZED AND NON-NORMALIZED VALUES FOR EACH EYE MOVEMENT FOR EACH WINDOW?????
#### AND TRY TO FIGURE OUT WHICH EYE MOVEMENT FEATURES ARE THE BEST PREDICTORS / FIT

#%%
network_mapping = {
     "17Networks_1": "Visual Central (Visual A)",
     "17Networks_2": "Visual Peripheral (Visual B)",
     "17Networks_3": "Somatomotor A",
     "17Networks_4": "Somatomotor B",
     "17Networks_5": "Dorsal Attention A",
     "17Networks_6": "Dorsal Attention B",
     "17Networks_7": "Salience / Ventral Attention A",
     "17Networks_8": "Salience / Ventral Attention B",
     "17Networks_9": "Limbic A",
     "17Networks_10": "Limbic B",
     "17Networks_11": "Control C",
     "17Networks_12": "Control A",
     "17Networks_13": "Control B",
     "17Networks_14": "Temporal Parietal",
     "17Networks_15": "Default C",
     "17Networks_16": "Default A",
     "17Networks_17": "Default B"
 }

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

processed_lfp_files = []  # initialize once


for vid in vids:
    
    if vid == 'despicable_me_english':
        #good_ET_pats = ["NS190"]#['NS140_02','NS153','NS164','NS166','NS174_02']#['NS127_02','NS135','NS136','NS137','NS138','NS140','NS140_02'] #'NS153','NS164','NS166','NS174_02',"NS178","NS190","NS191"]#,'NS193',"NS194","NS201",'NS205']
       # good_ET_pats = ['NS193',"NS194","NS201",'NS205']
       # good_ET_pats = ['NS127_02','NS135','NS136','NS137','NS138','NS140','NS151','NS153','NS154','NS155_02','NS164','NS174_02','NS174_03','NS190','NS191','NS193','NS194','NS178','NS201_02','NS204','NS205']
      # good_ET_pats = ['NS193'] 
       patients = [
         'NS127_02',
         'NS135',
         'NS136',
         'NS137',
         'NS138',
         'NS140',
         'NS140_02',
         'NS151',
         'NS153',
         'NS154',
         'NS155_02',
         'NS164',
         'NS174_02',
         'NS174_03',
         'NS178',
         'NS190',
         'NS191',
         'NS193',
         'NS194',
         'NS201_02',
         'NS204',
         'NS205']
       
    if vid == 'the_present':
        patients = [
         # 'NS135',
         'NS140',
         'NS137',
         'NS144_02',
         'NS149',
         'NS153',
         'NS154',
         'NS155',
         'NS155_02',
         'NS164',
         'NS174_03',
         'NS178',
         'NS190',
         'NS192',
         'NS193',
         'NS208',
         'NS210']
         #['NS174_03','NS210']

         
    elif vid == 'inscapes':
        patients = [
         'NS127_02',
         'NS135',
         'NS136',
         'NS137',
         'NS138',
         'NS140',
         'NS140_02',
         'NS151',
         'NS153',
         'NS155',
         'NS155_02',
         'NS164',
         'NS178',
         'NS205',
         'NS210']
        
    patients.sort()
        
    if vid == 'despicable_me_english':
        keys = ['despicable_me_english','dme']
    if vid == 'inscapes':
        keys = ['inscapes']
    if vid == 'the_present':
        keys = ['present','the_present']

    
    # Load ISC data 
    
    # if vid in ['dme', 'despicable_me_english']:
    #         data = np.load('/Volumes/Samsung/Movie_data/ISC_despicable_me_english_30Dec25/ISC_despicable_me_english_30Dec25_despicable_me_english_isc_gaze_position.npz',allow_pickle = True)
    # if vid == 'inscapes':
    #         data = np.load("/Volumes/Samsung/Movie_data/ISC_inscapes_30Dec25/ISC_inscapes_30Dec25_inscapes_isc_gaze_position.npz",allow_pickle = True)

    # #is time_isc time or isc...
    # time_isc = data['time_isc']
    # patients_isc = data['patients']
    # isc_time = data['isc_time_gaze']
    
    # fs_isc = 1 / np.mean(np.diff(time_isc))   
    
    # patients_isc = data['patients']
    
  #   freq_band_count = 0
    
#   for freq_band in freq_bands:
#       if freq_band == 'alpha':
#           freq_range = (8, 13)
#       elif freq_band == 'HFA':
#           freq_range = (62, 150)
#       elif freq_band == 'delta':
#           freq_range = (1, 3)
#       elif freq_band == 'theta':
#           freq_range = (4, 7)
#       elif freq_band == 'beta':
#           freq_range = (14, 30)
#       elif freq_band == 'gamma':
#           freq_range = (31, 59)
#       elif freq_band == 'all_gamma':
#           freq_range = (31,150)
#       elif freq_band == 'mid_gamma':
#           freq_range = (50,69)
#       elif freq_band == 'theta_alpha':
#           freq_range = (4,13)
#       elif freq_band == 'all':
#           freq_range = (0,170)
#       else:
#           raise Exception("no range assigned")
       
#       if freq_band == 'alpha':
#           bin_width = 2
#       elif freq_band == 'beta':
#           bin_width = 4
#       elif freq_band in ['gamma', 'all_gamma','mid_gamma']:
#           bin_width = 8
#       elif freq_band == 'HFA':
#           bin_width = 10
#       else:
#           bin_width = 2  # default
  
        
    print(f"Processing data for {vid}")

    fig_dir = f'/{machine_path}/Samsung/Movie_data/{output}_{vid}_all_cortContacts_tf_10s_{tf_freq_range}_21Apr26'
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)
      #
            #%
    for pat in patients:
        pat_dir = os.path.join(data_dir, pat)
        fig_patient_dir = os.path.join(fig_dir, pat)
        if not os.path.exists(fig_patient_dir):
            os.makedirs(fig_patient_dir)
                    
        lfp_pat_dir = '{:s}/{:s}/Neural_prep'.format(mne_data_dir, pat)
        
        lfp_files = os.listdir(lfp_pat_dir)
          
      #if pat == 'NS166':
     #     lfp_files = list(compress(lfp_files, [f'_{ref}_outliers' in f for f in lfp_files]))
     # else:
     # lfp_files = [f for f in lfp_files if ref in f and f.endswith('.fif')]
      
    #  vid_list = [f for f in lfp_files if any(k in f for k in keys)]
      
        lfp_files = [
            f for f in os.listdir(lfp_pat_dir)
            if f.endswith(".fif")
            and ref in f
            and ('referenced' in f)
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
            print(f"  Using: {os.path.basename(excel_path)}")
            elec_ref_table = pd.read_excel(excel_path)
            
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
        
                lfp = mne_data.get_data()
                fs_lfp = mne_data.info['sfreq']
                time_lfp = mne_data.times
                    
                lfp_ip = lfp[idx_ip, :]
                    
                labels_ip = list(compress(labels, idx_ip))


                f_start, f_end = tf_freq_range
                freqs_tf = np.arange(f_start, f_end + freq_step, freq_step)
                
                # Remove flat channels before processing (std < threshold)
                flat_std_thresh = 1e-6
                stds = np.std(lfp_ip, axis=1)
                nonflat_idx = stds > flat_std_thresh
                lfp_ip = lfp_ip[nonflat_idx]
                labels_ip = [label for i, label in enumerate(labels_ip) if nonflat_idx[i]]
                
                if lfp_ip.shape[0] == 0:
                    print(f"[SKIP] No non-flat intracranial channels remain for {entry_id}")
                    continue

                info = mne.create_info(ch_names=labels_ip,sfreq = fs_lfp)

                                        
                if output == 'wavelet':
                    data_tf = lfp_ip[np.newaxis, :, :]   # (1, n_channels, n_times)
                
                    if n_cycles_mode == 'scaled':
                        n_cycles = freqs_tf / 2.0
                    else:
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
                
                    power_tf = power_tf[0]   # (n_channels, n_freqs, n_times_decim)
                
                    if pow_type == 'log':
                        power_tf = np.log10(power_tf + 1e-6)
                
                    pow_tf_dat = power_tf
                    fs_tf = fs_lfp / decim_tf
                    t_tf = np.arange(pow_tf_dat.shape[2]) / fs_tf
                

                elif output == 'multitaper':
                    data_tf = lfp_ip[np.newaxis, :, :]
                
                    if n_cycles_mode == 'scaled':
                        n_cycles = freqs_tf / 2.0
                    else:
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
                
                    power_tf = power_tf[0]
                
                    if pow_type == 'log':
                        power_tf = np.log10(power_tf + 1e-6)
                
                    pow_tf_dat = power_tf
                    fs_tf = fs_lfp / decim_tf
                    t_tf = np.arange(pow_tf_dat.shape[2]) / fs_tf


                ## collapse to windows
                window_samples = int(window_length_sec * fs_tf)
                step_size_sec = window_length_sec - overlap_sec
                step_samples = int(step_size_sec * fs_tf)
                
                # windowed_tf = []
                # window_centers = []
                
                # for i in range(target_num_steps):
                #     start = i * step_samples
                #     end = start + window_samples
                #     if end > pow_tf_dat.shape[2]:
                #         break
                #     win_mean = np.mean(pow_tf_dat[:, :, start:end], axis=2)   # (n_channels, n_freqs)
                #     windowed_tf.append(win_mean)
                #     window_centers.append((start + end) / 2 / fs_tf)
                
                # windowed_tf = np.stack(windowed_tf, axis=0)   # (n_windows, n_channels, n_freqs)
                # window_centers = np.array(window_centers)
                
                from scipy.stats import trim_mean
                
                windowed_mean = []
                windowed_median = []
                windowed_trimmed = []
                windowed_std = []
                windowed_iqr = []
                windowed_n = []
                window_centers = []
                
                trim_prop = 0.10   # trims 10% from each tail
                
                for i in range(target_num_steps):
                    start = i * step_samples
                    end = start + window_samples
                    if end > pow_tf_dat.shape[2]:
                        break
                
                    win = pow_tf_dat[:, :, start:end]   # (n_channels, n_freqs, n_times_in_window)
                
                    win_mean = np.mean(win, axis=2)
                    win_median = np.median(win, axis=2)
                    win_trimmed = trim_mean(win, proportiontocut=trim_prop, axis=2)
                    win_std = np.std(win, axis=2)
                    win_iqr = np.percentile(win, 75, axis=2) - np.percentile(win, 25, axis=2)
                    win_n = np.full(win_mean.shape, win.shape[2], dtype=np.int32)
                
                    windowed_mean.append(win_mean)
                    windowed_median.append(win_median)
                    windowed_trimmed.append(win_trimmed)
                    windowed_std.append(win_std)
                    windowed_iqr.append(win_iqr)
                    windowed_n.append(win_n)
                
                    window_centers.append((start + end) / 2 / fs_tf)
                
                windowed_mean = np.stack(windowed_mean, axis=0)
                windowed_median = np.stack(windowed_median, axis=0)
                windowed_trimmed = np.stack(windowed_trimmed, axis=0)
                windowed_std = np.stack(windowed_std, axis=0)
                windowed_iqr = np.stack(windowed_iqr, axis=0)
                windowed_n = np.stack(windowed_n, axis=0)
                window_centers = np.array(window_centers)
                
                if len(windowed_mean) == 0:
                    print(f"[SKIP] No valid windows for {entry_id}")
                    continue
                           
                #
                tf_save_path = os.path.join(
                    fig_patient_dir,
                    f'{entry_id}_{vid}_{output}_{pow_type}_tf.npz'
                )
                                
                save_dict = {
                    'windowed_mean': windowed_mean,
                    'windowed_median': windowed_median,
                    'windowed_trimmed': windowed_trimmed,
                    'windowed_std': windowed_std,
                    'windowed_iqr': windowed_iqr,
                    'windowed_n': windowed_n,
                    'freqs_tf': freqs_tf,
                    'window_centers': window_centers,
                    'labels_ip': np.array(labels_ip, dtype=object),
                    'pat': pat,
                    'run_label': run_label,
                    'vid': vid,
                    'fs_tf': fs_tf,
                    'decim_tf': decim_tf
                }
                
                if save_continuous_tf:
                    save_dict['pow_tf_dat'] = pow_tf_dat
                    save_dict['t_tf'] = t_tf
                    
                
                np.savez_compressed(tf_save_path, **save_dict)
                print(f"Saved TF data to: {tf_save_path}")
                #%%

                # Initialize lists for DK_Atlas, Y7_Atlas, and AparcAseg_Atlas regions
                dk_regions = []
                y7_regions = []
                y17_regions = []
                aparc_aseg_regions = []
                
                for label in labels_ip:
                    match_row = elecs_subs[elecs_subs['label'] == label]
                
                    if not match_row.empty:
                        dk_col = 'DK_Atlas' if 'DK_Atlas' in elecs_subs.columns else 'Desikan_Killiany'
                        y7_col = 'Y7_Atlas' if 'Y7_Atlas' in elecs_subs.columns else 'Yeo7'
                        y17_col = 'Y17_Atlas' if 'Y17_Atlas' in elecs_subs.columns else 'Yeo17'
                        aparc_col = 'AparcAseg_Atlas' if 'AparcAseg_Atlas' in elecs_subs.columns else 'aparc_aseg'
                
                        dk_regions.append(match_row[dk_col].values[0])
                        y7_regions.append(match_row[y7_col].values[0])
                        y17_regions.append(match_row[y17_col].values[0])
                        aparc_aseg_regions.append(match_row[aparc_col].values[0])
                    else:
                        dk_regions.append('Unknown')
                        y7_regions.append('Unknown')
                        y17_regions.append('Unknown')
                        aparc_aseg_regions.append('Unknown')
                
                channel_meta = pd.DataFrame({
                    'label': labels_ip,
                    'DK_Atlas': dk_regions,
                    'Y7_Atlas': y7_regions,
                    'Y17_Atlas': y17_regions,
                    'AparcAseg_Atlas': aparc_aseg_regions
                })
                
                meta_csv = os.path.join(fig_patient_dir, f'{entry_id}_{vid}_channel_metadata.csv')
                channel_meta.to_csv(meta_csv, index=False)
                
                #%%
                
                import numpy as np
                
                tf_file = tf_save_path
                data = np.load(tf_file, allow_pickle=True)
                
                windowed_tf = data['windowed_tf']        # (windows, channels, freqs)
                freqs = data['freqs_tf']
                labels = data['labels_ip']
                window_centers = data['window_centers']
                
                print(windowed_tf.shape)
                print(freqs.shape)
                print(labels[:5])
                
                import matplotlib.pyplot as plt
                
                ch = 0  # pick a channel
                
                # ---------------------------
                # RAW SPECTROGRAM (with clipping)
                # ---------------------------
                spec = windowed_tf[:, ch, :]
                
                plt.figure(figsize=(8, 5))
                plt.imshow(
                    spec.T,
                    aspect='auto',
                    origin='lower',
                    extent=[window_centers[0], window_centers[-1], freqs[0], freqs[-1]],
                    vmin=np.percentile(spec, 5),
                    vmax=np.percentile(spec, 95)
                )
                
                plt.colorbar(label='Log Power')
                plt.xlabel('Time (s)')
                plt.ylabel('Frequency (Hz)')
                plt.title(f'Channel: {labels[ch]} (clipped){pat}')
                plt.show()
                
                
                # ---------------------------
                # Z-SCORED SPECTROGRAM (KEY ADDITION)
                # ---------------------------
                spec_z = (spec - spec.mean(axis=0, keepdims=True)) / spec.std(axis=0, keepdims=True)
                
                plt.figure(figsize=(8, 5))
                plt.imshow(
                    spec_z.T,
                    aspect='auto',
                    origin='lower',
                    extent=[window_centers[0], window_centers[-1], freqs[0], freqs[-1]]
                )
                
                plt.colorbar(label='Z-scored log power')
                plt.xlabel('Time (s)')
                plt.ylabel('Frequency (Hz)')
                plt.title(f'Channel: {labels[ch]} (z-scored) {pat}')
                plt.show()
                
                
                # ---------------------------
                # HIGH-FREQUENCY ONLY VIEW
                # ---------------------------
                hf_mask = freqs >= 30
                spec_hf = spec[:, hf_mask]
                freqs_hf = freqs[hf_mask]
                
                spec_hf_z = (spec_hf - spec_hf.mean(axis=0, keepdims=True)) / spec_hf.std(axis=0, keepdims=True)
                
                plt.figure(figsize=(8, 5))
                plt.imshow(
                    spec_hf_z.T,
                    aspect='auto',
                    origin='lower',
                    extent=[window_centers[0], window_centers[-1], freqs_hf[0], freqs_hf[-1]]
                )
                
                plt.colorbar(label='Z-scored log power')
                plt.xlabel('Time (s)')
                plt.ylabel('Frequency (Hz)')
                plt.title(f'{labels[ch]} (30–150 Hz) {pat}')
                plt.show()
                
                
                # ---------------------------
                # MEAN ACROSS CHANNELS
                # ---------------------------
                mean_tf = np.mean(windowed_tf, axis=1)  # (windows, freqs)
                
                plt.figure(figsize=(8, 5))
                plt.imshow(
                    mean_tf.T,
                    aspect='auto',
                    origin='lower',
                    extent=[window_centers[0], window_centers[-1], freqs[0], freqs[-1]],
                    vmin=np.percentile(mean_tf, 5),
                    vmax=np.percentile(mean_tf, 95)
                )
                
                plt.colorbar(label='Log Power')
                plt.xlabel('Time (s)')
                plt.ylabel('Frequency (Hz)')
                plt.title(f'Mean across channels {pat}')
                plt.show()
                
                
                # ---------------------------
                # GLOBAL POWER SPECTRUM
                # ---------------------------
                mean_freq_profile = np.mean(windowed_tf, axis=(0, 1))
                
                plt.figure()
                plt.plot(freqs, mean_freq_profile)
                plt.xlabel('Frequency (Hz)')
                plt.ylabel('Mean log power')
                plt.title(f'Global power spectrum {pat}')
                plt.show()
                
                
                # ---------------------------
                # HFA TIMECOURSE
                # ---------------------------
                band_mask = (freqs >= 70) & (freqs <= 150)
                
                hfa = np.mean(windowed_tf[:, :, band_mask], axis=2)  # (windows, channels)
                hfa_mean = np.mean(hfa, axis=1)
                
                plt.figure()
                plt.plot(window_centers, hfa_mean)
                plt.xlabel('Time (s)')
                plt.ylabel('HFA (log power)')
                plt.title(f'HFA timecourse {pat}')
                plt.show()
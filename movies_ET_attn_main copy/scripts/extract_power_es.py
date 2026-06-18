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
import multiprocessing

from scipy.signal import decimate
from scipy.signal import correlate

from joblib import Parallel, delayed
from fooof import FOOOF
from fooof.plts.spectra import plot_spectrum

machine_path = 'media/christine'#'Volumes' #'media/christine'


#from antropy import sample_entropy, spectral_entropy, perm_entropy, lziv_complexity

# Add Linux library paths BEFORE importing epipe
sys.path.insert(0, f'/{machine_path}/Samsung/EPIPE/Python')
sys.path.insert(0, f'/{machine_path}/Samsung/iEEG2NWB-main')

#vids = ['inscapes','despicable_me_english']#,'despicable_me_english']
vids = ['betta']#,'inscapes']#,'despicable_me_english']#,'despicable_me_english']
freq_bands = ['theta']#['delta','theta','alpha','gamma','HFA']#'beta','gamma','HFA'] #'delta','theta','alpha','beta','gamma'
#freq_bands = ['theta_alpha','all_gamma']

ref = 'avg'
region = 'all'


data_dir = f'/{machine_path}/Samsung/Movie_data/exp_samp_prep_standard'
isc_dir = f'/{machine_path}/SamsungMovie_data/data/isc'
mne_data_dir = f'/{machine_path}/Samsung/Movie_data/exp_samp_prep_standard'
elec_dir = f'/{machine_path}/Samsung/Movie_data/data/electrode_localization'
fs_dir = f'/{machine_path}/Samsung/anatomy'

fooof_dir = '/Volumes/Samsung/Movie_data/fooof_theta_peak_lookup_all.csv'

corr_dir = f'/{machine_path}/Samsung/Movie_data/data/movie_elec_corr_sheets'

fs_eye = 300

visualize_mne_steps = False
condense_to_isc = False
rolling_average = False
lowpass = False
find_peaks = True
plot_power = False
plot_power_subsets = False
use_interpolation = False
window_compare = False
extract_power = False

output = 'power'
pow_type = 'raw'

data_type = 'epoch'

entropy_type = 'mssd'

wd = '/Volumes/Samsung/scripts/movies_ET_attn_main'
src_dir = os.path.join(wd, 'src')
src_dir = os.path.abspath(src_dir)


if src_dir not in sys.path:
    sys.path.insert(0, src_dir)


from epipe import nwb2mne, inspectNwb
from epipe import inspectNwb, nwb2mne, read_ielvis
import sys

sys.path.insert(0, f'/{machine_path}/Samsung/scripts/movies_ET_attn_main/src')
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

#%% Helpers
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


def atlas_labels_for_channels(corr, pat, labels_ip, atlas_col="DK_Atlas"):
    atlas_df = corr.rename(columns={"SubID": "pat_base", "Contact": "electrode"}).copy()
    atlas_df["pat_base"] = atlas_df["pat_base"].astype(str).str.strip()
    atlas_df["electrode"] = atlas_df["electrode"].astype(str).str.strip()

    lab = pd.DataFrame({"electrode": [str(x).strip() for x in labels_ip]})
    lab["pat_base"] = str(pat).strip()

    lab = lab.merge(
        atlas_df[["pat_base", "electrode", "Hem", "DK_Atlas", "DK_Lobe", "Y7_Atlas", "Y17_Atlas", "AparcAseg_Atlas"]],
        on=["pat_base", "electrode"],
        how="left"
    )

    if atlas_col not in lab.columns:
        raise ValueError(f"atlas_col='{atlas_col}' not found. Available: {list(lab.columns)}")

    lab["atlas_label"] = lab[atlas_col].astype("string").fillna("Unknown").str.strip()

    # optional hemi standardization
    lab["hem"] = lab["Hem"].astype(str).str.upper().str.strip()
    lab.loc[lab["hem"].isin(["LEFT","LH"]), "hem"] = "L"
    lab.loc[lab["hem"].isin(["RIGHT","RH"]), "hem"] = "R"
    lab["hem"] = lab["hem"].where(lab["hem"].isin(["L","R"]), "")

    return lab


#%% Main script

vids.sort()
freq_bands.sort()

processed_lfp_files = []  # initialize once


for vid in vids:

       
    if vid == 'betta':
        patients = [
            'NS217'
            ]
         
    elif vid == 'inscapes':
        patients = [
         'NS217'
         ]
    
    
    patients.sort()
        
    if vid == 'betta':
        keys = ['betta','Betta']
    if vid == 'inscapes':
        keys = ['inscapes']


    
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
            fig_dir = f'/{machine_path}/Samsung/Movie_data/exp_samp_output/{output}_{pow_type}_{freq_band}_{vid}_18Jun26'
        if not os.path.exists(fig_dir):
            os.makedirs(fig_dir)
            
        if find_peaks == True:
            fooof_fig_dir = f'/{machine_path}/Samsung/Movie_data/exp_samp_output/es_fooof_peaks_{freq_band}_{vid}'

       #%%     
        for pat in patients:
            pat_dir = os.path.join(data_dir, pat)
            fig_patient_dir = os.path.join(fig_dir, pat)
            if not os.path.exists(fig_patient_dir):
                os.makedirs(fig_patient_dir)
                
            lfp_pat_dir = '{:s}/{:s}/Neural_prep'.format(mne_data_dir, pat)
            
            lfp_files = os.listdir(lfp_pat_dir)
                        
            
            lfp_files = [
                f for f in os.listdir(lfp_pat_dir)
                if f.endswith(".fif")
                and ref in f
                and ('referenced' in f)
                and data_type in f
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

                mne_data = mne.read_epochs(os.path.join(lfp_pat_dir, lfp_file), preload=False)

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

              
                ## ADD CHECK FOR BAD CHANNELS HERE, THOUGH NOT CURRENTLY NEEDED BECAUSE HAVE REMOVED BAD CHANS FROM EPOCHS .FIF
                
                # Visually inspect epoched data
                if visualize_mne_steps:

                    mne_data.plot(
                        title=f"{pat} - {vid} - Probe-onset epochs: -12.1 to -0.1 s",
                        scalings=dict(seeg=100e-6, ecog=100e-6),
                        n_channels=64,
                        show_scrollbars=True,
                        n_epochs = 1,
                        block=True
                    )
                       
                os.makedirs(lfp_pat_dir, exist_ok=True)
                
                labels = mne_data.ch_names
                                        
                exclude_strings = ['bankssts']
            
                ip_contacts = elecs_subs.label.values
            
            
                ## can filter included contacts by atlas label or other var from elec_subs 
                #ip_contacts = elecs_subs.Contact.values[(elecs_subs.AparcAseg_Atlas != 'Right-Cerebral-White-Matter') & (elecs_subs.AparcAseg_Atlas != 'Left-Cerebral-White-Matter')]
                #ip_contacts = elecs_subs.loc[elecs_subs['DK_Lobe'].isin(['Right-Hippocampus', 'Left-Hippocampus']),'Contact'].values
                 
                
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
 #%%                ## FIND PEAK FREQUENCY USING FOOOF           
                    fooof_region = None
    
                    if find_peaks:
                        if not os.path.exists(fooof_fig_dir):
                            os.makedirs(fooof_fig_dir)
                            
                        fooof_fig_patient_dir = os.path.join(fooof_fig_dir, pat)
                        if not os.path.exists(fooof_fig_patient_dir):
                            os.makedirs(fooof_fig_patient_dir)
                                
                        fooof_results = []
                   
                        # The band of interest is already defined earlier in the script
                        peak_search_range = freq_range
                    
                        # Broader fitting ranges for estimating aperiodic background + candidate peaks
                        fooof_fit_ranges = {
                            'delta': (1, 13),
                            'theta': (1, 13),
                            'alpha': (4, 30),
                            'beta':  (4, 40),
                            'gamma': (20, 80),
                            'HFA':   (30, 170),
                            'all_gamma': (20, 170),
                            'mid_gamma': (20, 90),
                            'theta_alpha': (1, 30),
                            'all': (1, 170),
                        }
                   
                        if freq_band not in fooof_fit_ranges:
                            raise ValueError(f"No FOOOF fit range defined for freq_band={freq_band}")
                            
                        if freq_band in ['gamma', 'HFA', 'all_gamma', 'mid_gamma']:
                            peak_width_limits = [2, 20]
                        else:
                            peak_width_limits = [1, 6]
                            
                        fooof_fmin, fooof_fmax = fooof_fit_ranges[freq_band]
                   
                        # Compute PSD over the broader FOOOF fitting range
                        psd_fooof, freqs_fooof = psd_array_welch(
                            lfp_ip,
                            sfreq=fs_lfp,
                            fmin=fooof_fmin,
                            fmax=fooof_fmax,
                            n_fft=int(fs_lfp * 2),
                            n_overlap=int(fs_lfp),
                            average='mean'
                        )
                    
                        for ch_idx, label in enumerate(labels_ip):
         
                            fm = FOOOF(
                                peak_width_limits=peak_width_limits,
                                max_n_peaks=6,
                                min_peak_height=0.1,
                                aperiodic_mode='fixed',
                                verbose=False
                            )
                                
                            try:
                                fm.fit(freqs_fooof, psd_fooof[ch_idx, :])
                    
                                all_peaks = fm.peak_params_
                    
                                if all_peaks.shape[0] > 0:
                                    in_band = (
                                        (all_peaks[:, 0] >= peak_search_range[0]) &
                                        (all_peaks[:, 0] <= peak_search_range[1])
                                    )
                    
                                    band_peaks = all_peaks[in_band]
                    
                                    if band_peaks.shape[0] > 0:
                                        # Pick the strongest in-band peak
                                        best_peak = band_peaks[np.argmax(band_peaks[:, 1])]
                    
                                        peak_freq = best_peak[0]
                                        peak_amp = best_peak[1]
                                        peak_width = best_peak[2]
                                        found_peak = True
                                        n_peaks_total = all_peaks.shape[0]
                                        n_peaks_in_band = band_peaks.shape[0]
                                    else:
                                        peak_freq = np.nan
                                        peak_amp = np.nan
                                        peak_width = np.nan
                                        found_peak = False
                                        n_peaks_total = all_peaks.shape[0]
                                        n_peaks_in_band = 0
                                else:
                                    peak_freq = np.nan
                                    peak_amp = np.nan
                                    peak_width = np.nan
                                    found_peak = False
                                    n_peaks_total = 0
                                    n_peaks_in_band = 0
                    
                                r_squared = fm.r_squared_
                                fit_error = fm.error_
                    
                            except Exception as e:
                                peak_freq = np.nan
                                peak_amp = np.nan
                                peak_width = np.nan
                                found_peak = False
                                n_peaks_total = np.nan
                                n_peaks_in_band = np.nan
                                r_squared = np.nan
                                fit_error = np.nan
                                print(f"FOOOF failed for {pat} {run_label} {label}: {e}")
                    
                            # Look up atlas region
                            fooof_region = None
                            match_row = elecs_subs[elecs_subs['label'] == label]
                    
                            if not match_row.empty:
                                if 'Y17_Atlas' in match_row.columns:
                                    fooof_region = match_row['Y17_Atlas'].values[0]
                                elif 'Y7_Atlas' in match_row.columns:
                                    fooof_region = match_row['Y7_Atlas'].values[0]
                    
                            fooof_results.append({
                                'Patient': pat,
                                'Run': run_label,
                                'Video': vid,
                                'FreqBand': freq_band,
                                'Channel': label,
                                'Region': fooof_region,
                    
                                'FOOOF_Fit_Range_Low': fooof_fmin,
                                'FOOOF_Fit_Range_High': fooof_fmax,
                                'Peak_Search_Range_Low': peak_search_range[0],
                                'Peak_Search_Range_High': peak_search_range[1],
                    
                                'Found_Peak': found_peak,
                                'Peak_Frequency': peak_freq,
                                'Peak_Amplitude': peak_amp,
                                'Peak_Width': peak_width,
                    
                                'N_Peaks_Total': n_peaks_total,
                                'N_Peaks_In_Band': n_peaks_in_band,
                                'FOOOF_R2': r_squared,
                                'FOOOF_Error': fit_error
                            })
                    
                        fooof_df = pd.DataFrame(fooof_results)
                    
                        fooof_csv = os.path.join(
                            fooof_fig_patient_dir,
                            f'{pat}_{vid}_{run_label}_{region}_{freq_band}_fooof_peaks.csv'
                        )
                    
                        fooof_df.to_csv(fooof_csv, index=False)
                        print(f'Saved FOOOF peak results to {fooof_csv}')
                        
                        if not extract_power:
                            continue
 #%%                   
                    if plot_power:
                        # recreate MNE Info object
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
                        
                    if output == "power":
                        
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
                        pow_dat_raw = np.mean(
                            np.stack(raw_power_bins, axis=0),
                            axis=0
                        )
                    
                        if pow_type == "raw":
                            pow_dat = pow_dat_raw
                        elif pow_type == "log":
                            pow_dat = np.log10(pow_dat_raw + 1e-6)
                        else:
                            raise ValueError(f"Unknown pow_type: {pow_type}")

                        
                        t_lfp = np.arange(lfp_ip.shape[2]) / fs_lfp
               

                        if plot_power:
                            # pow_dat should be shape:
                            # n_epochs x n_channels x n_times
                        
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
                                

                # ------------------------------------------------------------
                # Save pow_ip and labels
                # ------------------------------------------------------------
                
                pow_ip = pow_dat          # shape: n_epochs x n_channels x n_times
                labels = labels_ip
                
                out_dir = os.path.join(lfp_pat_dir, "power_outputs")
                os.makedirs(out_dir, exist_ok=True)
                
                out_fname = f"{pat}_{vid}_{ref}_{freq_band}_analytic_power_epochs.npz"
                out_path = os.path.join(out_dir, out_fname)
                
                np.savez(
                    out_path,
                    pow_ip=pow_ip,
                    labels=np.asarray(labels),
                    fs_lfp=fs_lfp,
                    t_lfp=t_lfp,
                    freq_bins=np.asarray(freq_bins),
                    pow_type=pow_type,
                    pat=pat,
                    vid=vid,
                    ref=ref,
                )
                
                print(f"Saved power data to: {out_path}")
                print(f"pow_ip shape: {pow_ip.shape}")   # n_epochs x n_channels x n_times
                print(f"n labels: {len(labels)}")

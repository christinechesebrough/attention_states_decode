#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 20 12:15:12 2024

@author: christinechesebrough
"""
#%%
import os, sys, re
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"

import numpy as np
import mne
from pynwb import NWBHDF5IO
import pandas as pd
import matplotlib

import matplotlib.pyplot as plt
#import cv2
import scipy.interpolate as interp
import scipy.stats as stats

import scipy.signal as signal
#import temporal_response_function as trf
from mne.time_frequency import psd_array_welch
from mne.filter import filter_data
from scipy.io.wavfile import write
import importlib.util
import glob
from scipy.signal import hilbert
from collections import defaultdict

#mne.viz.set_browser_backend('qt')

#matplotlib.use("qt")

machine_path = 'media/christine'#'media/christine'#'Volumes' #'media/christine'


sys.path.insert(0, f'/{machine_path}/Samsung/EPIPE-movie_nwb/Python')
sys.path.append(f'/{machine_path}/Samsung/iEEG2NWB-main')

# sys.path.insert(0, '/Users/christinechesebrough/Documents/EPIPE-movie_nwb/Python')
# sys.path.append('/Users/christinechesebrough/Documents/iEEG2NWB-main')

import pycircstat2
from pycircstat2 import hypothesis
from epipe import inspectNwb, nwb2mne, read_ielvis, reref_avg, reref_bipolar,filter_hfa_continuous, reref_white_matter

wd = f'/{machine_path}/Samsung/scripts/movies_ET_attn_main'
src_dir = os.path.join(wd, 'src')
src_dir = os.path.abspath(src_dir)

# Add src to path if not already there
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

# Peek inside src to confirm the file is really there and spelled exactly
print('src contents (first 20):')
print([os.path.basename(p) for p in glob.glob(os.path.join(src_dir, '*'))][:20])

# 1) Put src on sys.path (at the front)
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)
print('on sys.path?', src_dir in sys.path)

import eeg_preproc_helpers

# Import EEG preprocessing helper functions
from eeg_preproc_helpers import (
    plot_power_spectra,  plot_psd_batched, plot_psd_with_scales,
    create_file_paths, apply_highpass_filter,
    save_bad_channels, load_bad_channels, check_bad_channels_integrity, load_preprocessed_data, validate_mne_structure, preserve_mne_structure, restore_mne_structure, load_data_with_bad_channels, summarize_bad_channels, 
    interpolate_spikes, make_groups_from_prefix, regress_out_noise_by_group,
    detect_spikes_all_channels, reref_avg_by_group,
    ProcessingLogger, detect_spikes_ref1
)

# path to temporal response function
module_path = '/Users/christinechesebrough/Documents/data_standardization-main/src/temporal_response_function.py'
module_name = 'temporal_response_function'

# spec = importlib.util.spec_from_file_location(module_name, module_path)
# trf = importlib.util.module_from_spec(spec)
# spec.loader.exec_module(trf)

def electrode_group(ch: str) -> str:
    return re.sub(r'\d+$', '', ch)

def pick_ptd_column(elecs_subs: pd.DataFrame) -> str | None:
    for col in elecs_subs.columns:
        if 'ptd' in col.lower():
            return col
    return None

def compute_line_noise_ratio(psd, freqs, mains=60.0, tol=1.0):
    # ratio: power around mains+harmonics divided by broadband power
    broadband = psd[(freqs >= 1) & (freqs <= 150)].mean()
    ln_bins = []
    for h in [mains, 2*mains, 3*mains]:
        ln_bins.append(psd[(freqs >= h - tol) & (freqs <= h + tol)].mean())
    line = np.nanmean(ln_bins)
    return float(line / (broadband + 1e-12))

def make_wm_metrics_df(ecogPreproc, wm_contacts, elecs_subs, fs_data, fmin=1, fmax=150):
    # Clean candidates
    wm_clean = [ch for ch in wm_contacts
                if ch in ecogPreproc.ch_names and ch not in ecogPreproc.info['bads']]
    if len(wm_clean) == 0:
        raise RuntimeError("No WM contacts remain after excluding bads / missing channels.")

    # Extract PTD values
    ptd_col = pick_ptd_column(elecs_subs)
    if ptd_col is None:
        raise RuntimeError("No PTD column found in elecs_subs.")

    ptd_map = dict(zip(elecs_subs['Label'], elecs_subs[ptd_col]))

    # Data
    wm_data = ecogPreproc.get_data(picks=wm_clean)
    wm_psd, freqs = psd_array_welch(wm_data, sfreq=fs_data, fmin=fmin, fmax=fmax, n_fft=int(fs_data * 2))

    # Metrics per channel
    mean_psd = wm_psd.mean(axis=1)
    variance = wm_data.var(axis=1)

    line_ratio = []
    for i in range(wm_psd.shape[0]):
        line_ratio.append(compute_line_noise_ratio(wm_psd[i], freqs))

    df = pd.DataFrame({
        'Contact': wm_clean,
        'Group': [electrode_group(ch) for ch in wm_clean],
        'Mean_PSD': mean_psd,
        'Variance': variance,
        'LineNoiseRatio': line_ratio,
        'PTD': [ptd_map.get(ch, np.nan) for ch in wm_clean],
    })

    return df


def robust_z(x):
    med = np.nanmedian(x)
    mad = np.nanmedian(np.abs(x - med)) + 1e-12
    return (x - med) / (1.4826 * mad)

def auto_select_wm_refs(wm_df, *,
                        ptd_thresh=-0.8,
                        max_per_group=1,
                        max_total=12,
                        var_floor_quantile=0.02):
    df = wm_df.copy()

    # Filter to WM-like contacts and those with PTD present
    df = df[df['PTD'].notna()]
    df = df[df['PTD'] < ptd_thresh].copy()

    if len(df) == 0:
        raise RuntimeError("No WM candidates pass PTD threshold.")

    # Avoid 'dead' contacts: extremely low variance can be suspicious
   # var_floor = df['Variance'].quantile(var_floor_quantile)
   # df = df[df['Variance'] > var_floor].copy()

    # Robust z-scoring: lower is better for PSD, variance, line noise
    z_psd = robust_z(df['Mean_PSD'].values)
    z_var = robust_z(df['Variance'].values)
    z_ln  = robust_z(df['LineNoiseRatio'].values)

    # For PTD, "more negative is better" (more WM). Convert so lower is better:
    # e.g., distance to -1
    ptd_dist = np.abs(df['PTD'].values - (-1.0))
    z_ptd = robust_z(ptd_dist)

    # Composite score (tweak weights as you like)
    df['Score'] = (0.20*z_psd + 0.20*z_var +  0.40*z_ptd)

    # Select top N per group
    selected = (
        df.sort_values('Score')
          .groupby('Group', group_keys=False)
          .head(max_per_group)
          .sort_values('Score')
    )

    # Cap total
    selected = selected.head(max_total)

    return selected


def plot_stft_spectrogram_raw(
    raw,
    ch_names,
    fmin=80,
    fmax=130,
    tmin=None,
    tmax=None,
    win_sec=1.0,
    overlap=0.75,
    db=True
):
    """
    STFT spectrogram for Raw data (no averaging), plotted for each channel.
    """

    # Crop if desired (recommended for very long recordings)
    raw_use = raw.copy()
    if (tmin is not None) or (tmax is not None):
        raw_use.crop(tmin=tmin, tmax=tmax)

    sfreq = raw_use.info["sfreq"]
    data, times = raw_use.get_data(picks=ch_names, return_times=True)

    nperseg = int(round(win_sec * sfreq))
    noverlap = int(round(overlap * nperseg))

    for i, ch in enumerate(ch_names):
        x = data[i]

        f, t, Sxx = spectrogram(
            x,
            fs=sfreq,
            nperseg=nperseg,
            noverlap=noverlap,
            scaling="density",
            mode="psd"
        )

        keep = (f >= fmin) & (f <= fmax)
        f2 = f[keep]
        S2 = Sxx[keep, :]

        # Convert to dB for visibility
        if db:
            Splot = 10 * np.log10(S2 + 1e-20)
        else:
            Splot = S2

        plt.figure(figsize=(10, 4))
        plt.pcolormesh(t + (tmin or 0), f2, Splot, shading="auto")
        plt.ylim(fmin, fmax)
        plt.xlabel("Time (s)")
        plt.ylabel("Frequency (Hz)")
        plt.title(f"Spectrogram (STFT, no averaging) — {ch}")
        plt.colorbar(label="PSD (dB)" if db else "PSD")
        plt.tight_layout()
        plt.show()


# Parameters
full_task_name = 'movies'
pipeline_name = 'preprocess_movies'
pipeline_version = 'v.1.0.725'

resample_fs = 600

vid = 'Betta'#["dme", "despicable_me_english"]

# Types of references to use in analyses
# Must be a list containing at least one of the options: "avg", "bip"
ref_types = ['avg']

n_jobs = 16

convert_db = True

# Frequency range
freq_range = [70, 150]
n_freq_bins = 10
freq_space = 'log'      # 'log', 'lin'

resample_bha_fs = 100

freq_bands = ['low','middle','high']

movie_table = pd.read_excel(f'/{machine_path}/Samsung/Movie_data/data/electrode_localization/movie_table.xlsx')

# Define directories

#data_dir = f'/{machine_path}/Samsung/Movie_data/movies_nwb_standard'
data_dir = f'/{machine_path}/Samsung/exp_sampling/es_nwb_standard'
#data_dir = f'/{machine_path}/Samsung/Movie_data/movies_new_nwb'
#data_dir = f'/{machine_path}/Samsung/AV40_data/new_converted'
#data_dir = f'/{machine_path}/Samsung/exp_samp_data/converted'
fs_dir = f'/{machine_path}/Samsung/anatomy'
#prep_dir = f'/{machine_path}/Samsung/AV40_data/new_converted'
prep_dir = f'/{machine_path}/Samsung/Movie_data/exp_samp_prep_standard'
corr_dir = f'/{machine_path}/Samsung/Movie_data/data/movie_elec_corr_sheets'

frame_dir = f'/{machine_path}/Samsung/Movie_data/data/video_frames'
lum_dir = f'/{machine_path}/Samsung/Movie_data/data/luminance'

if not os.path.exists(lum_dir):
    os.makedirs(lum_dir)

compute_luminance = False

et_prep_dir = 'Eye_prep'
audio_dir = 'Audio'
neural_prep_dir = 'Neural_prep'
hfa_dir = 'HFA'

# Single File Processing Mode
# Set to True to process only one specific file
single_file_mode = True 

# When single_file_mode is True, specify the file to process
# Format: 'patient_id' or 'patient_id_implant_number'
target_patient = 'NS217'#'NS205'  # e.g., 'NS189' or 'NS189_01'
#target_movie = 'sub-NS127_ses-02_task-the_present_run-1_ieeg.nwb'
target_movie = 'NS217_ses-Exp_Samp_Betta01_behavior+ecephys.nwb'

patients = ["NS211"]#,"NS190","NS191","NS193","NS194","NS201_02","NS204","NS205"]


#%% Initialize loop and open NWB

broken_nwb = []

if single_file_mode:
    # Single file processing mode
    print(f"\n{'='*60}")
    print(f"SINGLE FILE PROCESSING MODE")
    print(f"Target Patient: {target_patient}")
    print(f"Target Movie: {target_movie}")
    print(f"{'='*60}\n")
    
    # Extract session number from the target movie filename
    # Expected format: NS189_ses-02_task-despicable_me_english_run-01_ieeg.nwb
    if 'ses-0' in target_movie:
        # Extract session number from filename
        ses_match = re.search(r'ses-(\d+)', target_movie)
        if ses_match:
            ses_num = ses_match.group(1)
            imp = f'ses-{ses_num}'
        else:
            print(f"❌ ERROR: Could not extract session number from filename: {target_movie}")
            #sys.exit(1)
    else:
        # Fallback to ses-01 if no session info in filename
        imp = 'ses-01'
    
    # Set patient path
    pat = f'{target_patient}'

    # Validate that the target file exists
    if data_dir == f'/{machine_path}/Samsung/Movie_data/movies_nwb_standard':
        target_nwb_path = '{:s}/{:s}/{:s}/{:s}'.format(data_dir, pat, imp, target_movie)
    elif data_dir == f'/{machine_path}/Samsung/movie_data_new/movies_new_nwb_fall25':
        target_nwb_path = '{:s}/{:s}/{:s}'.format(data_dir, pat, target_movie)
    elif data_dir == f'/{machine_path}/Samsung/Movie_data/movies_new_nwb':
        target_nwb_path = '{:s}/{:s}/{:s}/ieeg/{:s}'.format(data_dir, pat, imp, target_movie)
    elif data_dir ==f'/{machine_path}/Samsung/AV40_data/new_converted':
        target_nwb_path = '{:s}/{:s}/{:s}/ieeg/{:s}'.format(data_dir, pat, imp, target_movie)
    elif data_dir == f'/{machine_path}/Samsung/exp_samp_data/converted':
        target_nwb_path = '{:s}/{:s}/{:s}/ieeg/{:s}'.format(data_dir, pat, imp, target_movie)
    elif data_dir == f'/{machine_path}/Samsung/exp_sampling/es_nwb_standard':
        target_nwb_path = '{:s}/{:s}/{:s}/ieeg/{:s}'.format(data_dir, pat, imp, target_movie)

    if not os.path.exists(target_nwb_path):
        print(f"❌ ERROR: Target file not found: {target_nwb_path}")
        print("Please check the target_patient and target_movie parameters.")
        print(f"Expected path: {target_nwb_path}")
        #sys.exit(1)
    
    print(f"✅ Target file found: {target_nwb_path}")
    
    # Process single file
    patients_to_process = [pat]
    implants_to_process = [imp]
    movies_to_process = [target_movie]
    
else:
    # Batch processing mode
    print(f"\n{'='*60}")
    print(f"BATCH PROCESSING MODE")
    print(f"Patients to process: {patients}")
    print(f"{'='*60}\n")
    
    patients_to_process = [f'{p}' for p in patients]
    implants_to_process = []
    movies_to_process = []

# Process files
    
implants = os.listdir('{:s}/{:s}'.format(data_dir, pat))
   
if single_file_mode:
    # Single file mode - use predefined implants and movies
    implants_to_process = implants_to_process
    movies_to_process = movies_to_process
else:
    # Batch mode - get all implants and movies
    implants_to_process = implants
    movies_to_process = []

for imp in implants_to_process:
    
    if data_dir == f'/{machine_path}/Samsung/Movie_data/movies_nwb_standard':
        ieeg_dir = '{:s}/{:s}/{:s}'.format(data_dir, pat, imp)
    elif data_dir == f'/{machine_path}/Samsung/movie_data_new/movies_new_nwb_fall25':
        ieeg_dir = '{:s}/{:s}/'.format(data_dir, pat)
    elif data_dir == f'/{machine_path}/Samsung/Movie_data/movies_new_nwb':
        ieeg_dir = '{:s}/{:s}/{:s}'.format(data_dir, pat, imp)
    elif data_dir ==f'/{machine_path}/Samsung/AV40_data/new_converted':
        ieeg_dir = '{:s}/{:s}/{:s}'.format(data_dir, pat, imp)
    elif data_dir == f'/{machine_path}/Samsung/exp_samp_data/converted':
        ieeg_dir = '{:s}/{:s}/{:s}'.format(data_dir, pat, imp)
    elif data_dir == f'/{machine_path}/Samsung/exp_sampling/es_nwb_standard':
        ieeg_dir = '{:s}/{:s}/{:s}'.format(data_dir, pat, imp)



    if single_file_mode:
        # Single file mode - use predefined movie
        movies = movies_to_process
    else:
        # Batch mode - get all movies
        movies = [f for f in os.listdir(ieeg_dir) if f.endswith('.nwb')]
        # Filter for specific video if vid is defined
        if 'vid' in locals():
            vid_keys = [vid] if isinstance(vid, str) else list(vid)
            movies = [
                mov for mov in movies
                if any(k in mov for k in vid_keys)
            ]
    
    # Extract session number from implant name (e.g., 'ses-01' -> 1)
    ses_match = re.search(r'ses-(\d+)', imp)
    if ses_match:
        ses_num = int(ses_match.group(1))
    else:
        ses_num = 1  # Default to session 1 if no match
    
    if ses_num == 1:
        pat_fs = pat.replace('sub-', '')
    else:
        pat_fs = '{:s}_{:02d}'.format(pat.replace('sub-', ''), ses_num)

    sub_fs_dir = '{:s}/{:s}'.format(fs_dir, pat_fs)
    
    sub_et_prep_dir = '{:s}/{:s}/{:s}'.format(prep_dir, pat_fs, et_prep_dir)

    for mov in movies:
        # Create standardized file paths for this movie
        file_paths = create_file_paths(
            patient_id=pat_fs,
            implant_id=imp,
            movie_filename=mov,
            prep_dir=prep_dir,
            neural_prep_dir=neural_prep_dir,
            hfa_dir=hfa_dir
        )
        
        # Initialize logger for this movie
        movie_name = file_paths['movie_base']
        logger = ProcessingLogger(pat_fs, movie_name, file_paths['sub_prep_dir'])
        
        # Construct the NWB file path
        if data_dir == f'/{machine_path}/Samsung/Movie_data/movies_nwb_standard':
            nwb_fname = '{:s}/{:s}/{:s}/{:s}'.format(data_dir, pat, imp, mov)
        elif data_dir == f'/{machine_path}/Samsung/movie_data_new/movies_new_nwb_fall25':
            nwb_fname = '{:s}/{:s}/{:s}'.format(data_dir, pat, mov)
        elif data_dir == f'/{machine_path}/Samsung/Movie_data/movies_new_nwb':
            nwb_fname = '{:s}/{:s}/{:s}/ieeg/{:s}'.format(data_dir, pat, imp, mov)
        elif data_dir ==f'/{machine_path}/Samsung/AV40_data/new_converted':
            nwb_fname = '{:s}/{:s}/{:s}/ieeg/{:s}'.format(data_dir, pat, imp,mov)
        elif data_dir == f'/{machine_path}/Samsung/exp_samp_data/converted':
            nwb_fname = '{:s}/{:s}/{:s}/ieeg/{:s}'.format(data_dir, pat, imp,mov)
        elif data_dir ==f'/{machine_path}/Samsung/exp_sampling/es_nwb_standard':
            nwb_fname = '{:s}/{:s}/{:s}/ieeg/{:s}'.format(data_dir, pat, imp, mov)


        # proceed
        print(f"\n{'='*60}")
        print(f"PROCESSING FILE:")
        print(f"Patient: {pat_fs}")
        print(f"Implant: {imp}")
        print(f"Movie: {mov}")
        print(f"File: {nwb_fname}")
        print(f"{'='*60}\n")
        
        if single_file_mode:
            print("⚠️ SINGLE FILE MODE: Processing only this file")
            print("⚠️ Make sure this is the correct file before proceeding!\n")
            
            # Ask for confirmation in single file mode
            confirm = input("Continue with this file? (y/n): ")
            if confirm.lower() != 'y':
                print("Processing cancelled.")
                sys.exit(0)
            print("✅ Proceeding with processing...\n")
        
        # Log file information
        logger.log_decision("FILE", f"Processing: {nwb_fname}")
        logger.log_decision("PATIENT", f"Patient: {pat_fs}, Implant: {imp}")
#%%
#
        # NWB read
        io = NWBHDF5IO(nwb_fname, mode='r', load_namespaces=True)

        nwb = io.read()
        # Log NWB file loading
     #   logger.log_nwb_load(nwb_fname)
        
        # Get info on data in NWB file
        nwbInfo = inspectNwb(nwb)
        tsInfo = nwbInfo['timeseries']
        elecTable = nwbInfo['elecs']
        
        # Define the subject-specific directory
        subid = os.path.basename(sub_fs_dir)
        elecReconDir = os.path.join(sub_fs_dir, 'elec_recon')
        
        elec_recon_dir = corr_dir    

        excel_files = sorted([
            f for f in os.listdir(elec_recon_dir)
            if pat in f
            and f.endswith('.xlsx')
            and not f.startswith('.')
        ])
        if not excel_files:
            print(f"  [SKIP] No correspondence .xlsx for {pat} found in {elec_recon_dir}")
            #continue

        # Use most recently modified if multiple exist (matches Python HFO script)
        excel_path = max(
            [os.path.join(elec_recon_dir, f) for f in excel_files],
            key=os.path.getmtime
        )
        print(f"  Using: {os.path.basename(excel_path)}")
        elec_ref_table = pd.read_excel(excel_path)
        
        # # List all files in the directory and filter for the one containing 'correspondence'
        # excel_files = [f for f in os.listdir(elecReconDir)
        #                if 'correspondence' in f and
        #                f.endswith('.xlsx') and
        #                not f.startswith('.')]
        
        # if not excel_files:
        #     print("No correspondence Excel file found.")
        
        # elif len(excel_files) == 1:
        #     excel_file = os.path.join(elecReconDir, excel_files[0])
        #     print(f"Found electrode correspondence file: {excel_file}")
        
        # else:
        #     print("Multiple correspondence Excel files found:")
        #     for i, f in enumerate(excel_files):
        #         print(f"  [{i}] {f}")
        
        #     while True:
        #         try:
        #             idx = int(input("Select the correct file by index: "))
        #             if 0 <= idx < len(excel_files):
        #                 excel_file = os.path.join(elecReconDir, excel_files[idx])
        #                 print(f"Selected electrode correspondence file: {excel_file}")
        #                 break
        #             else:
        #                 print("Index out of range. Try again.")
        #         except ValueError:
        #             print("Please enter a valid integer.")

    
        # Load into Pandas DataFrame
        #elec_ref_table = pd.read_excel(excel_file)
        
        print(elec_ref_table.head())  # Preview the data
        #ADD ERROR HANDLING HERE
        #print("No correspondence Excel file found in", elecReconDir)

        # Get ieeg data
            
        if 'ieeg' in tsInfo['name'].to_list():
            ecogContainer = nwb.acquisition.get('ieeg')
            fs = ecogContainer.rate
            ecog = nwb2mne(ecogContainer,preload=False)
            
        
            # Get coordinates of each electrode
            try:
                ielvis_df = read_ielvis(sub_fs_dir)
                ch_coords = {}
                nan_array = np.empty((3,)) * np.nan
                for thisChn in ecog.ch_names:
                    idx = np.where(ielvis_df['label'] == thisChn)[0]
                    if len(idx) == 1:
                        xyz = np.array(ielvis_df.iloc[idx[0]]['LEPTO'])
                        ch_coords[thisChn] = xyz/1000
                    elif len(idx) == 0:
                        ch_coords[thisChn] = nan_array
                    else:
                        raise ValueError('More than 1 found!')
            except:
                ch_coords = {}
                for thisChn in ecog.ch_names:
                    ch_coords[thisChn] = np.empty((3,)) * np.nan
        
            # Create `montage` data structure as required by MNE1
            montage = mne.channels.make_dig_montage(ch_pos=ch_coords, coord_frame='mri')
            montage.add_estimated_fiducials(pat_fs, fs_dir)
            ecog.set_montage(montage)
            
            #mont = mne.channels.make_dig_montage( ch_pos={ch: np.array([np.nan, np.nan, np.nan]) for ch in raw_ekg.ch_names}, coord_frame="mri", )
           # raw_ekg = nwb2mne(nwb.acquisition["ekg"], preload=False, create_montage=False)
           # raw_ekg.set_channel_types({ch: "ecg" for ch in raw_ekg.ch_names})
            
            # ecog.add_channels([raw_ekg], force_update_info=True)
            
            # single montage call, but allow missing (EKG)
           # ecog.set_montage(montage, on_missing="ignore")
            
            # Validate MNE structure after montage setup
            validate_mne_structure(ecog, "After Montage Setup")
            
            # Load audio
            audioContainer = nwb.acquisition.get('audio')
            fs_audio = audioContainer.rate
            audio = audioContainer.data[:]
            t_audio = np.arange(0, audio.shape[0]) / fs_audio
            
        #except:
            
            #broken_nwb.append(nwb_fname)
       # continue
        
        # Get the current sampling rate. Important for later
        orig_fs = ecog.info['sfreq']
        
        # Get the TTL pulses. Specify the name of the container with the TTL pulses
        ttl_container_name = 'TTL'
        try:
            ttls = nwb.get_acquisition(ttl_container_name).timestamps[()]
        except:
            # An analog TTL channel, convert to discrete timestamps
            ana_ttls = nwb.get_acquisition(ttl_container_name).data[()].flatten()
            ttl_rate = nwb.get_acquisition(ttl_container_name).rate
            from epipe import ana2dig
            _, ttls = ana2dig(ana_ttls, fs=ttl_rate, min_diff=0.4, return_time=True)
        
        ttl_id = nwb.acquisition['TTL'].data[:]
       # 
       
        # Preprocess the EEG data
        # Create preprocessing directory using standardized paths
        if not os.path.exists(file_paths['sub_prep_dir']):
            os.makedirs(file_paths['sub_prep_dir'])

        ecog.plot(
            scalings=dict(seeg=200e-6),  
            n_channels=32,               
            remove_dc=False,              # remove mean per channel
            show_scrollbars=True,        # allow vertical scrolling
            duration=12.0,
            block = True# time window in seconds
        )
        
        # raw_ekg.plot(
        #     scalings=dict(seeg=200e-6),  
        #     n_channels=32,               
        #     remove_dc=False,              # remove mean per channel
        #     show_scrollbars=True,        # allow vertical scrolling
        #     duration=12.0,
        #     block = True# time window in seconds
        # )

    #%%    
 #       THIS IS OPTIONAL BUT SOMETIMES NECESSARY; IT'S APPARENT WHETHER IT IS NECESSARY BASED ON THE NEXT STEP OF NOTCH FILTERING, BANDPASS, AND DOWNSAMPLING
 ## If bad initial reference, option to rereference before notch filtering and downsample
 
        reref_to_ref = False
        if reref_to_ref:
            ref_channels = [ch for ch in ecog.ch_names if ch.lower().startswith('ref')]
            print("Candidate reference channels:", ref_channels)
            
            for ref in ref_channels:
                data = ecog.copy().pick(ref).get_data()
                std = np.std(data)
                print(f"{ref}: std={std}")
                            
            # Ask for user input by channel name
            selected_ref = input("Enter the NAME of the best reference channel to use (case-sensitive): ")
            
            if selected_ref in ref_channels:
                print(f"Selected reference channel: {selected_ref}")
                
                # Apply referencing
                ecog.set_eeg_reference(ref_channels=[selected_ref], projection=False)
                print("Re-referencing applied with selected channel.")
            else:
                print("Invalid channel name. No referencing applied.")
                
                
                #%%
        # #Notch filter, bandpass filter, then downsample

        print('--->Applying notch filters, bandpass filter, and downsampling to %2.fHz' % resample_fs)
        
        # Log filtering section
       # logger.log_section("FILTERING STEPS")
        
        # Apply notch filter first to remove line noise
        if pat == "NS135":
            notch_freqs = (60,104,120,180)
        # if pat == 'NS178':
        #     notch_freqs = (58,60,62,118,120,122,178,180,182)
        if pat == "NS151": 
            notch_freqs = (60,104,118,180)
        elif pat == "NS166":
            #notch_freqs = (57,60,67,114,120,134,180)
            notch_freqs = (57, 60, 67,68, 84, 111,112,120, 128, 135, 167, 180)
        # elif pat == "NS211":
        #     notch_freqs = (57,60,67,114,120,134,180)
        elif pat == "NS174":
            if nwb_fname == f'/{machine_path}/Samsung/Movie_data/movies_nwb_standard/NS174/ses-03/sub-NS174_ses-03_task-despicable_me_english_run-01_ieeg.nwb':
                notch_freqs = (60,120,143,180)
        else:    
            notch_freqs = (60, 120, 180)
            
        ecog.notch_filter(freqs=notch_freqs, notch_widths=2)
      #  logger.log_filtering("NOTCH", f"Frequencies: {notch_freqs} Hz, Widths: 2 Hz")
        
        #
        # Apply bandpass filter 
        ecog.filter(l_freq=0.1, h_freq=170.0)
       # logger.log_filtering("BANDPASS", f"Low: 0.1 Hz, High: 170.0 Hz")
        
        # Then downsample to target sampling rate
        ecogResampled = ecog.resample(resample_fs)
       # logger.log_filtering("DOWNSAMPLING", f"From: {orig_fs} Hz, To: {resample_fs} Hz")
        
        # Validate structure after resampling
        validate_mne_structure(ecogResampled, "After Resampling")
    
        ecogResampled.crop(tmin=frame_time[0], tmax=frame_time[-1])
        
        # Validate structure after cropping
        validate_mne_structure(ecogResampled, "After Cropping")
        

#%% Remove spikes if necessary
            remove_spikes = False
            if remove_spikes:          
                spikes_by_chan = detect_spikes_all_channels(
                    ecogResampled,
                    picks=None,
                    thresh_z=5.5,
                    max_width_ms=20.0,
                    min_separation_ms=5.0
                    )
            
                interp_half_window_ms = 7.5
                sfreq = ecogResampled.info['sfreq']
                interp_half_window_samp = int(interp_half_window_ms * sfreq / 1000.0)
                
                raw_clean = ecogResampled.copy()
                data = raw_clean._data
                n_ch, n_samp = data.shape
                
                for ch_idx, ch_name in enumerate(raw_clean.ch_names):
                    spike_samples = spikes_by_chan.get(ch_name, np.array([], dtype=int))
                    if len(spike_samples) == 0:
                        continue
                
                    print(f"Interpolating {len(spike_samples)} spikes in channel {ch_name}")
                
                    for s in spike_samples:
                        seg_start = max(1, s - interp_half_window_samp)
                        seg_end   = min(n_samp - 2, s + interp_half_window_samp)
                
                        left = seg_start - 1
                        right = seg_end + 1
                
                        seg_idx = np.arange(seg_start, seg_end + 1)
                        data[ch_idx, seg_idx] = np.interp(
                            seg_idx,
                            [left, right],
                            data[ch_idx, [left, right]]
                        )
                
                ecogResampled = raw_clean
                
#%%         # Display the raw traces and mark bad channels
        print("\n--- Manual Bad Channel Marking ---")
        print("Mark bad channels in the interactive plot, then close the plot to continue.")
        fig = ecogResampled.plot(
            scalings=dict(seeg=100e-6),
            n_channels=64,
            remove_dc=True,
            show_scrollbars=True,
            duration=10.0,
            block=True,  
            )
        
        # Now you are guaranteed the plot is closed and bads are updated (if your marking took effect).
        summarize_bad_channels(ecogResampled, "After Manual Marking (post-close)")
        save_success = save_bad_channels(
            ecogResampled,
            
            file_paths['bad_channels_file'],
            pat_fs,
            file_paths['movie_base']
        )
                    
        # Check bad channel integrity after manual marking
        integrity_check = check_bad_channels_integrity(ecogResampled, "After Manual Marking")
        
        # Print bad channel status for debugging
        summarize_bad_channels(ecogResampled, "After Manual Marking")
        print(f"Bad channels saved to: {file_paths['bad_channels_file']}")
        
        # Log bad channel selection
        logger.log_section("BAD CHANNEL SELECTION")
        logger.log_bad_channels(ecogResampled.info['bads'])
        logger.log_file_save("BAD_CHANNELS", file_paths['bad_channels_file'])
        
        # Preprocessed file name after downsampling and bad channel removal
        preprocessed_filename = file_paths['preprocessed_file']
        
        # Save the current state of the data in the MNE format                                
        ecogPreproc = ecogResampled

        # Save preprocessed data after downsampling and bad channel removal
        ecogResampled.save(preprocessed_filename, fmt='single', overwrite=True)
        print(f"Preprocessed data saved to: {preprocessed_filename}")
        logger.log_fif_save(preprocessed_filename, file_description="Preprocessed FIF")
        
        # Final integrity check before saving
        final_integrity = check_bad_channels_integrity(ecogPreproc, "Final Preprocessed Data")
        
        # Final MNE structure validation
        final_structure = validate_mne_structure(ecogPreproc, "Final Preprocessed Data")


    

#%% reload if necessary
            reload = True
            
            if reload:
                
                preprocessed_filename = file_paths['preprocessed_file']
                fif_file = preprocessed_filename
                print(f"reloading cut and filtered data:{preprocessed_filename}")
                
                ecogPreproc = mne.io.read_raw_fif(fif_file, preload=True)
                
                # Print basic info summary
                print(ecogPreproc)
                
                # Print channel names
                print("\nChannel names:")
                print(ecogPreproc.ch_names)
                
                # Plot to visually inspect data tracesf
                ecogPreproc.plot(
                    scalings=dict(seeg=200e-6),
                    n_channels=32,
                    remove_dc=True,
                    decim=10,
                    show_scrollbars=True,   # test
                    duration=20.0,           # test
                )


#%%  # Visualize PSD 
            data = ecogPreproc.get_data()  # Shape: (n_channels, n_samples)

            fs_data = resample_fs
            
            # Get good channel indices
            labels = list(ecogPreproc.ch_names)
            good_ch_indices = [i for i, ch in enumerate(labels) if ch not in ecogPreproc.info['bads']]
            
            # Filter the data to include only good channels
            data_good = data[good_ch_indices, :]
            labels_good = [labels[i] for i in good_ch_indices]
            
            # Inspect power spectra at each frequency band
            for freq_band in freq_bands:
                if freq_band == 'low':
                    freq_range = (0.5, 7)
                elif freq_band == 'middle':
                    freq_range = (8, 50)
                elif freq_band == 'high':
                    freq_range = (70, 150)
                elif freq_band == "all":
                    freq_range = (1,170)
                
                # Bandpass filter only the good channels
                sos = signal.butter(5, list(freq_range), btype='bandpass', output='sos', fs=fs_data)
                band_dat = signal.sosfiltfilt(sos, data_good, axis=1)
            
                # Plot power spectra excluding bad channels
                plot_power_spectra(
                    band_dat, labels_good, fs_data, freq_range, pat, 
                    plot_title=f"Power Spectral Density (Welch) for {freq_band}" #for {pat}
                )
                  
            #%% OPTIONAL
            compute_welch = False
            if compute_welch:              
                # Compute PSD using Welch's method
                psds, freqs = psd_array_welch(
                    data_good, sfreq=fs_data, fmin=0.5, fmax=170, 
                    n_fft=int(fs_data * 2), n_overlap=int(fs_data), average='mean'
                )
                
                # Convert PSD to dB scale
                psds_db = 10 * np.log10(psds)
                
                # Compute median and standard deviation of PSD across channels
                median_psd = np.median(psds_db, axis=0)
                z_scores = (psds_db - median_psd) / np.std(psds_db, axis=0)
                
                # Identify bad channels based on z-score threshold
                bad_threshold = 5  # Adjust threshold as needed
                bad_channel_indices = np.where(np.abs(z_scores).max(axis=1) > bad_threshold)[0]
                bad_channels = [labels_good[i] for i in bad_channel_indices]
                
                print(f"Identified {len(bad_channels)} bad channels: {bad_channels}")

#%%  #      #Visualize PSD in smaller batches for readability
                
            print("\n" + "="*60)
            print("NEW INTERACTIVE PSD VISUALIZATION")
            print("="*60)
            
            #Subset visualization for manageable overview
            print("\n--- Option 1: PSD Subset Visualization ---")
            print("Creating manageable subset plots (20 channels)...")
            
            # Create batched PSD plots using the original plotting style (32 channels = 2 electrodes)
            batched_figures = plot_psd_with_scales(
                ecogPreproc, 
                scale_type='log_y',  # Log x-axis, linear y-axis #log_y #log_x
                batch_size=32,
                freq_bands=['high'],
                patient_id=pat)

            #%%   # Find white matter electrodes to use as reference montage
            
            if 'wm_legacy' in ref_types:
            #
                # Check if a white matter contact file exists for this patient and movie
                wm_contact_file = file_paths['wm_contacts_file']
                 
                elecs_subs = elec_ref_table
                            
                if os.path.exists(wm_contact_file):
                    # Load the previously saved WM contacts
                    with open(wm_contact_file, 'r') as f:
                        wm_references = [line.strip() for line in f.readlines()]
                    print(f"Loaded white matter references from file: {wm_references}")
                    selected_wm_references = wm_references
                
                else:
                    # Use the electrode reference table to find white matter electrodes to use as reference montage
                
                    # Check if a white matter contact file exists for this patient and movie
                    wm_contact_file = file_paths['wm_contacts_file']
                    elecs_subs = elec_ref_table
                    
    
                    # Define white matter group labels
                    wm_group = ['Right-Cerebral-White-Matter', 'Left-Cerebral-White-Matter']
                
                    # Determine WM contacts based on available atlas columns
                    if 'FS_vol' in elecs_subs.columns:
                        wm_mask = elecs_subs['FS_vol'].isin(wm_group)
                        wm_contacts = elecs_subs.loc[wm_mask, 'label'].values
                    elif 'AparcAseg_Atlas' in elecs_subs.columns:
                        wm_mask = elecs_subs['AparcAseg_Atlas'].isin(wm_group)
                        wm_contacts = elecs_subs.loc[wm_mask, 'Contact'].values
                    elif 'aparc_aseg' in elecs_subs.columns:
                        wm_mask = elecs_subs['aparc_aseg'].isin(wm_group)
                        wm_contacts = elecs_subs.loc[wm_mask, 'label'].values
                    else:
                        wm_contacts = []
                        print("⚠️ No appropriate atlas column found for white matter extraction.")
                
                    print(f"Identified white matter contacts: {wm_contacts}")
                
                    # Exclude bad contacts
                    wm_contacts_cleaned = [ch for ch in wm_contacts if ch not in ecogPreproc.info['bads']]
                    print(f"White matter contacts (cleaned): {wm_contacts_cleaned}")
                
                    # Extract data for cleaned WM contacts
                    wm_data = ecogPreproc.get_data(picks=wm_contacts_cleaned)
                
                    # Compute PSD
                    wm_psd, freqs = psd_array_welch(
                        wm_data, sfreq=fs_data, fmin=1, fmax=150, n_fft=int(fs_data * 2)
                    )
                
                    # Calculate mean PSD and variance per contact
                    mean_psd = wm_psd.mean(axis=1)
                    variance = wm_data.var(axis=1)
                
                    # Extract PTD_index values and DK Atlas labels for each contact
                    dk_labels = []
                    for ch in wm_contacts_cleaned:            
                        if 'Desikan_Killiany' in elecs_subs.columns:
                            atlas_label = elecs_subs.loc[elecs_subs['label'] == ch, 'Desikan_Killiany']
                        elif 'DK_Atlas' in elecs_subs.columns:
                            atlas_label = elecs_subs.loc[elecs_subs['label'] == ch, 'DK_Atlas']
                        else:
                            atlas_label = pd.Series(['Unknown'])
                    
                        dk_labels.append(atlas_label.values[0] if not atlas_label.empty else 'Unknown')
                    
                    # Extract PTD_index values and DK Atlas labels for each contact
                    ptd_values= []
                    
                    for col in elecs_subs.columns:
                        if 'ptd' in col.lower():
                            print(col)
                            ptd_col = col
                            break
                    
                    if ptd_col is None:
                        print("⚠️ No PTD column found.")
                        ptd_values = [np.nan] * len(wm_contacts_cleaned)
                    else:
                        # Extract PTD_index values using the identified column
                        ptd_values = []
                        for ch in wm_contacts_cleaned:
                            ptd_row = elecs_subs.loc[elecs_subs['label'] == ch, ptd_col]
                            ptd_values.append(ptd_row.values[0] if not ptd_row.empty else np.nan)
                            
                    dk_labels = []
                    for ch in wm_contacts_cleaned:
                        if 'Desikan_Killiany' in elecs_subs.columns:
                            atlas_label = elecs_subs.loc[elecs_subs['label'] == ch, 'Desikan_Killiany']
                        elif 'DK_Atlas' in elecs_subs.columns:
                            atlas_label = elecs_subs.loc[elecs_subs['label'] == ch, 'DK_Atlas']
                        else:
                            atlas_label = pd.Series(['Unknown'])
                
                        dk_labels.append(atlas_label.values[0] if not atlas_label.empty else 'Unknown')
                
                
                    # Construct DataFrame
                    wm_df = pd.DataFrame({
                        'Contact': wm_contacts_cleaned,
                        'Mean_PSD': mean_psd,
                        'Variance': variance,
                        'PTD_index': ptd_values,
                        'DK_Atlas_Label': dk_labels
                    })
                    
                    # Define output CSV filename in the same directory as wm_contact_file
                    wm_df_csv_file = wm_contact_file.replace('_wm_contacts.txt', '_wm_contacts_metrics.csv')
                    
                    # Save wm_df to CSV
                    wm_df.to_csv(wm_df_csv_file, index=False)
                    print(f"WM contact metrics saved to: {wm_df_csv_file}")
    
     
                    # Top 10 by lowest variance
                    wm_df_sorted_var = wm_df.sort_values(by='Variance')
                    print("\n📊 Top 10 contacts by lowest variance:")
                    for i, (_, row) in enumerate(wm_df_sorted_var.head(10).iterrows()):
                        print(f"  {i+1:2d}. {row['Contact']:8s} - PSD: {row['Mean_PSD']:8.2f}, Var: {row['Variance']:8.2e}, PTD: {row['PTD_index']:6.3f}")
                    
                    # Top 10 by PTD closest to -1
                    wm_df_sorted_ptd = wm_df.sort_values(by='PTD_index')
                    print("\n📊 Top 10 contacts by PTD closest to -1:")
                    for i, (_, row) in enumerate(wm_df_sorted_ptd.head(10).iterrows()):
                        print(f"  {i+1:2d}. {row['Contact']:8s} - PSD: {row['Mean_PSD']:8.2f}, Var: {row['Variance']:8.2e}, PTD: {row['PTD_index']:6.3f}")
                    
                    # Create composite score for best contacts
                    # Normalize each metric to 0-1 scale (lower is better)
                    wm_df['PSD_rank'] = wm_df['Mean_PSD'].rank()
                    wm_df['Var_rank'] = wm_df['Variance'].rank()
                    wm_df['PTD_rank'] = wm_df['PTD_index'].rank()
                    
                    # Composite score (lower is better)
                    wm_df['Composite_score'] = (wm_df['PSD_rank'] + wm_df['Var_rank'] + wm_df['PTD_rank']) / 3
                    
                    # Top 15 by composite score
                    wm_df_sorted_composite = wm_df.sort_values(by='Composite_score')
                    print("\n" + "="*60)
                    print("LOW VARIANCE /PSD WM CONTACTS (COMPOSITE SCORE)")
                    print("="*60)
                    print("Rank | Contact | PSD Rank | Var Rank | PTD Rank | Composite | PSD    | Variance | PTD")
                    print("-" * 80)
                    
                    for i, (_, row) in enumerate(wm_df_sorted_composite.head(15).iterrows()):
                        print(f" {i+1:2d}  | {row['Contact']:7s} | {row['PSD_rank']:8.1f} | {row['Var_rank']:8.1f} | {row['PTD_rank']:8.1f} | {row['Composite_score']:9.1f} | {row['Mean_PSD']:6.2f} | {row['Variance']:8.2e} | {row['PTD_index']:5.3f}")
                    
                    print("\n")
                    print("   These contacts have the best combination of low PSD, low variance, and PTD close to -1.")
    
                    # Plot WM contacts for visual inspection BEFORE user prompt
                    print("\n📈 Plotting WM contacts for visual inspection...")
                    fig = ecogPreproc.plot(
                        scalings=dict(seeg=200e-6),
                        n_channels=32,
                        remove_dc=True,
                        show_scrollbars=True,
                        duration=12.0
                    )
                    
                    # Wait for user to close the plot
                    plt.show(block=True)
                    
                    #%%
                
                    # Ask for user input to define good WM contacts by label
                    print("\n Select a subset of wm contacts from different electrodes/spatial locations")
    
                    selected_labels = input(
                        "\nEnter the LABELS of good WM contacts (comma-separated, e.g., RIp12,LFp4,LDh8): "
                    )
                    selected_wm_references = [label.strip() for label in selected_labels.split(',')]
                
                    # Verify entered labels are valid
                    valid_selected_wm_references = [ch for ch in selected_wm_references if ch in wm_df['Contact'].values]
                    invalid_labels = [ch for ch in selected_wm_references if ch not in wm_df['Contact'].values]
                    if invalid_labels:
                        print(f"⚠️ These entered labels were not found and will be skipped: {invalid_labels}")
                
                    print(f"Final selected WM references: {valid_selected_wm_references}")
                
                    # Save to file
                    with open(wm_contact_file, 'w') as f:
                        f.writelines("\n".join(valid_selected_wm_references))
                    print(f"Selected WM references saved to: {wm_contact_file}")
    
    #%%
                # Use the selected WM contacts for referencing
                wm_references = selected_wm_references
                print(f"Selected white matter references: {wm_references}")
#%%
            fs_data = resample_fs
            wm_contact_file = file_paths['wm_contacts_file']
        
            if 'wm' in ref_types:
                
                if os.path.exists(wm_contact_file):
                    # Load the previously saved WM contacts
                    with open(wm_contact_file, 'r') as f:
                        wm_references = [line.strip() for line in f.readlines()]
                    print(f"Loaded white matter references from file: {wm_references}")
                    wm_references_auto = wm_references
                
                else:
                    # Use the electrode reference table to find white matter electrodes to use as reference montage
                
                    # Check if a white matter contact file exists for this patient and movie
                    wm_contact_file = file_paths['wm_contacts_file']
                    elecs_subs = elec_ref_table
                    
        
                    # Define white matter group labels
                    wm_group = ['Right-Cerebral-White-Matter', 'Left-Cerebral-White-Matter']
                
                    # Determine WM contacts based on available atlas columns
                    if 'FS_vol' in elecs_subs.columns:
                        wm_mask = elecs_subs['FS_vol'].isin(wm_group)
                        wm_contacts = elecs_subs.loc[wm_mask, 'Label'].values
                    elif 'AparcAseg_Atlas' in elecs_subs.columns:
                        wm_mask = elecs_subs['AparcAseg_Atlas'].isin(wm_group)
                        wm_contacts = elecs_subs.loc[wm_mask, 'Contact'].values
                    elif 'aparc_aseg' in elecs_subs.columns:
                        wm_mask = elecs_subs['aparc_aseg'].isin(wm_group)
                        wm_contacts = elecs_subs.loc[wm_mask, 'Label'].values
                    else:
                        wm_contacts = []
                        print("⚠️ No appropriate atlas column found for white matter extraction.")
                
                    print(f"Identified white matter contacts: {wm_contacts}")
                
                    wm_df = make_wm_metrics_df(ecogPreproc, wm_contacts, elecs_subs, fs_data)
                    selected_df = auto_select_wm_refs(wm_df, ptd_thresh=-0.8, max_per_group=1, max_total=12)
                    
                    wm_references_auto = selected_df['Contact'].tolist()
                    print("Auto-selected WM references:", wm_references_auto)
                    
                    raw_wm = ecogPreproc.copy().pick_channels(wm_references_auto, ordered=True)
                    raw_wm.plot(
                        scalings=dict(seeg=200e-6),
                        n_channels=min(32, len(wm_references_auto)),
                        remove_dc=True,
                        duration=12.0,
                        show_scrollbars=True,
                        block = True
                    )
        
                    wm_references_auto = [
                        ch for ch in wm_references_auto
                        if ch not in raw_wm.info['bads']
                    ]
        
                    ecogPreproc.plot(
                        picks=wm_references_auto,
                        scalings=dict(seeg=200e-6, ecog=200e-6),
                        n_channels=min(32, len(wm_references_auto)),
                        remove_dc=True,
                        duration=12.0,
                        show_scrollbars=True
                    )
                    # Save to file
                    with open(wm_contact_file, 'w') as f:
                        f.writelines("\n".join(wm_references_auto))
                    print(f"Selected WM references saved to: {wm_contact_file}")

    
#%%
            # Apply referencing:
            logger.log_section("REFERENCING STEPS")
            
            # Create reference-specific bad channel files
            ref_bad_channels_files = {}
            
            for ref in ref_types:
                referenced_filename = file_paths['referenced_files'][ref]
                
                # Create reference-specific bad channels file
                ref_bad_channels_file = file_paths['bad_channels_file'].replace('_bad_channels.txt', f'_bad_channels_{ref}.txt')
                ref_bad_channels_files[ref] = ref_bad_channels_file
                
                # Copy original bad channels to reference-specific file
                if os.path.exists(file_paths['bad_channels_file']):
                    import shutil
                    shutil.copy2(file_paths['bad_channels_file'], ref_bad_channels_file)
                    print(f"Copied original bad channels to: {ref_bad_channels_file}")
                else:
                    # Create empty bad channels file for this reference
                    with open(ref_bad_channels_file, 'w') as f:
                        f.write(f"# Bad channels for {pat_fs} - {file_paths['movie_base']} - {ref} reference\n")
                        f.write(f"# Total channels: {len(ecogPreproc.ch_names)}\n")
                        f.write("# Bad channels: 0\n")
                        f.write("# Channel names:\n")
                    print(f"Created new bad channels file: {ref_bad_channels_file}")
            
                if ref == 'wm_legacy':
                    # Apply WM referencing
                    ecog_reref = ecogPreproc.set_eeg_reference(ref_channels=wm_references)
                    ecog_reref.save(referenced_filename, overwrite=True)
                    print(f"WM-referenced data saved to: {referenced_filename}")
                    logger.log_referencing("WM", f"Reference channels: {wm_references}")
                    logger.log_file_save("WM_REFERENCED", referenced_filename)
                    logger.log_fif_save(referenced_filename, file_description="WM Referenced FIF")
            
                if ref == 'wm':
                    # Apply WM referencing
                    #ecog_reref = ecogPreproc.set_eeg_reference(ref_channels=wm_references_auto)
 
                    raw_for_reref = ecogPreproc.copy().drop_channels(ecogPreproc.info["bads"])

                    ecog_reref, ref_data = mne.set_eeg_reference(
                        raw_for_reref,
                        ref_channels=wm_references_auto,
                        ch_type="seeg",
                    )

                    #ecogWMavg, _ = set_eeg_reference(ecogPreproc, ref_channels=wm_references_auto, copy=True)
                    ecog_reref.save(referenced_filename, overwrite=True)
                    print(f"WM-referenced data saved to: {referenced_filename}")
                    logger.log_referencing("WM", f"Reference channels: {wm_references_auto}")
                    logger.log_file_save("WM_REFERENCED", referenced_filename)
                    logger.log_fif_save(referenced_filename, file_description="WM Referenced FIF")
            
                elif ref == 'avg':
                    # Apply average referencing
                    if pat in ['NS211','NS144','NS128']:
                        elec_groups = make_groups_from_prefix(ecogPreproc)
                        ecog_reref = reref_avg_by_group(ecogPreproc, elec_groups)
                    else:
                        ecog_reref = reref_avg(ecogPreproc)
                    ecog_reref.save(referenced_filename, overwrite=True)
                    print(f"Average-referenced data saved to: {referenced_filename}")
                    logger.log_referencing("AVERAGE", "Using all good channels")
                    logger.log_file_save("AVG_REFERENCED", referenced_filename)
                    logger.log_fif_save(referenced_filename, file_description="Average Referenced FIF")
        
                elif ref == 'bip':
                    # Apply bipolar referencing (implement custom logic if needed)
                    ecog_reref = reref_bipolar(ecogPreproc)  # Example logic
                    ecog_reref.save(referenced_filename, overwrite=True)
                    print(f"Bipolar-referenced data saved to: {referenced_filename}")
                    logger.log_referencing("BIPOLAR", "Bipolar referencing applied")
                    logger.log_file_save("BIP_REFERENCED", referenced_filename)
                    logger.log_fif_save(referenced_filename, file_description="Bipolar Referenced FIF")
            
            #%% Visual inspection and bad channel marking for each reference type
            print("\n" + "="*60)
            print("REFERENCE-SPECIFIC BAD CHANNEL MARKING")
            print("="*60)
            print("Each reference type may have different bad channels.")
            print("Mark bad channels specific to each reference type.")
            print("="*60)
            
            for ref in ref_types:
                referenced_filename = file_paths['referenced_files'][ref]
                ref_bad_channels_file = ref_bad_channels_files[ref]
                
                print(f"\n--- Processing {ref.upper()} Reference ---")
                print(f"Referenced file: {referenced_filename}")
                print(f"Bad channels file: {ref_bad_channels_file}")
                
                # Load data with reference-specific bad channels
                ecog_reref = load_data_with_bad_channels(
                    referenced_filename,
                    ref_bad_channels_file, 
                    pat_fs, 
                    file_paths['movie_base']
                )
                
                # Display current bad channel status
                total_channels = len(ecog_reref.ch_names)
                current_bad = len(ecog_reref.info['bads'])
                good_channels = total_channels - current_bad
                
                print(f" Current status for {ref} reference:")
                print(f"  Total channels: {total_channels}")
                print(f"  Current bad channels: {current_bad}")
                print(f"  Good channels: {good_channels}")
                print(f"  Good channel percentage: {(good_channels/total_channels)*100:.1f}%")
                
                # Interactive plot for bad channel marking
                print(f"\n Interactive bad channel marking for {ref} reference")
                print("Mark bad channels in the plot, then close to continue...")
                
                fig = ecog_reref.plot(
                    title=f"{pat_fs} - {ref.upper()} Referenced Data (Mark Bad Channels)",
                    scalings=dict(seeg=100e-6),
                    n_channels=64,
                    remove_dc=True,
                    show_scrollbars=True,
                    duration=15.0,
                    show=True,
                    block=True
                )
                
                # Check bad channel integrity after marking
                integrity_check = check_bad_channels_integrity(ecog_reref, f"After {ref} Reference Marking")
                
                # Save reference-specific bad channels
                save_success = save_bad_channels(
                    ecog_reref, 
                    ref_bad_channels_file, 
                    pat_fs, 
                    f"{file_paths['movie_base']}_{ref}"
                )
                
                # Print updated bad channel status
                summarize_bad_channels(ecog_reref, f"After {ref} Reference Marking")
                print(f"Bad channels saved to: {ref_bad_channels_file}")
                
                # Log reference-specific bad channel selection
                logger.log_section(f"{ref.upper()} REFERENCE BAD CHANNELS")
                logger.log_bad_channels(ecog_reref.info['bads'], f"{ref} reference")
                logger.log_file_save(f"{ref.upper()}_BAD_CHANNELS", ref_bad_channels_file)
                #%%
                # Optional: PSD visualization for this reference
                print(f"\n📊 PSD visualization for {ref} reference")
                psd_figures = plot_psd_with_scales(
                    ecog_reref, 
                    scale_type='log_y',
                    batch_size=16,
                    freq_bands=['high'],
                    patient_id=f"{pat_fs}_{ref}"
                )
                
                print(f"✅ Completed bad channel marking for {ref} reference")
                print(f"  - Bad channels: {len(ecog_reref.info['bads'])}")
                print(f"  - Good channels: {len(ecog_reref.ch_names) - len(ecog_reref.info['bads'])}")
                print(f"  - Saved to: {ref_bad_channels_file}")


# %%

## CALCULATE TTL DIFFS
## SAVE TTL DF WITH FIF
## EPOCH 12 second windows in .FIF AT 'probe_onset' pulses, -12s to -.1 s to probe onset
            

# %%
            
            ## CALCULATE TTL DIFFS
            ## SAVE TTL DF WITH FIF AS ANNOTATIONS
            ## EPOCH -12.1 TO -0.1 s PRIOR TO probe_onset
            
            # ------------------------------------------------------------
            # 1. Make TTL dataframe using timing relative to raw start = 0
            # ------------------------------------------------------------
                        
            
            ttls = np.asarray(ttls, dtype=float)
            
            diff_from_prev = np.insert(np.diff(ttls), 0, np.nan)
            
            ttl_df = pd.DataFrame({
                "pulse_idx": np.arange(len(ttls)),
                "ttl_time_s": ttls,
                "diff_from_prev_s": diff_from_prev,
                "label": "other"
            })
            
            # First TTL = start
            ttl_df.loc[0, "label"] = "start"
            
            # Last TTL = end
            ttl_df.loc[len(ttl_df) - 1, "label"] = "end"
            
            # ------------------------------------------------------------
            # 2. Default candidate probe_onsets
            # ------------------------------------------------------------
            
            probe_mask = ttl_df["diff_from_prev_s"] > 30
            
            # ------------------------------------------------------------
            # 3. Patient/video-specific probe filters
            # ------------------------------------------------------------
            
            if pat == "NS217" and vid == "Betta":
                first_valid_probe = 172.1393152
            
                # Do not label any candidate probes before this time
                probe_mask = probe_mask & (ttl_df["ttl_time_s"] >= first_valid_probe)
            
            # ------------------------------------------------------------
            # 4. Assign final labels
            # ------------------------------------------------------------
            
            ttl_df.loc[probe_mask, "label"] = "probe_onset"
            
            # Preserve start/end labels in case they overlap with probe rule
            ttl_df.loc[0, "label"] = "start"
            ttl_df.loc[len(ttl_df) - 1, "label"] = "end"
            
            print(ttl_df)
            print(ttl_df["label"].value_counts())
            
            #%%
            # ------------------------------------------------------------
            # 3. Add TTL labels to Raw FIF as annotations
            # ------------------------------------------------------------
            # IMPORTANT:
            # We are treating ttl_time_s as relative to the beginning of ecog_reref.
            # Therefore, do NOT subtract ecog_reref.first_time.
            
            annotation_onsets = ttl_df["ttl_time_s"].values
            
            valid_annot_mask = (
                (annotation_onsets >= 0) &
                (annotation_onsets <= ecog_reref.times[-1])
            )
            
            if valid_annot_mask.sum() < len(ttl_df):
                print(
                    f"WARNING: {len(ttl_df) - valid_annot_mask.sum()} TTL annotations "
                    "fall outside the current raw duration and will be skipped."
                )
            
            ttl_annotations = mne.Annotations(
                onset=annotation_onsets[valid_annot_mask],
                duration=np.zeros(valid_annot_mask.sum()),
                description=ttl_df.loc[valid_annot_mask, "label"].values
            )
            
            ecog_reref.set_annotations(ttl_annotations)
            
            annotated_filename = referenced_filename.replace(".fif", "_with_ttl-raw.fif")
            ecog_reref.save(annotated_filename, fmt="single", overwrite=True)
            
            print(f"Annotated FIF with TTL labels saved to: {annotated_filename}")
            logger.log_fif_save(
                annotated_filename,
                file_description="Referenced FIF with TTL annotations"
            )
            
            fig = ecog_reref.plot(
                title=f"{pat_fs} - {ref.upper()} Referenced Data (Mark Bad Channels)",
                scalings=dict(seeg=100e-6),
                n_channels=64,
                remove_dc=True,
                show_scrollbars=True,
                duration=15.0,
                show=True,
                block=True
            )
            
        
            
            
            #%%
            # ------------------------------------------------------------
            # 4. Epoch -12.1 to -0.1 s before probe onset
            # ------------------------------------------------------------
            
            probe_df = ttl_df[ttl_df["label"] == "probe_onset"].copy()
            
            sfreq = ecog_reref.info["sfreq"]
            
            # Again: ttl_time_s is treated as relative to raw start = 0
            probe_onsets_s = probe_df["ttl_time_s"].values
            probe_samples = np.round(probe_onsets_s * sfreq).astype(int)
            
            tmin = -12.1
            tmax = -0.1
            
            epoch_start_samples = probe_samples + int(np.round(tmin * sfreq))
            epoch_end_samples = probe_samples + int(np.round(tmax * sfreq))
            
            valid_epoch_mask = (
                (epoch_start_samples >= 0) &
                (epoch_end_samples < ecog_reref.n_times)
            )
            
            if valid_epoch_mask.sum() < len(probe_df):
                print(
                    f"WARNING: {len(probe_df) - valid_epoch_mask.sum()} probe epochs "
                    "were skipped because the full -12.1 to -0.1 s window did not fit "
                    "inside the raw recording."
                )
            
            probe_samples_valid = probe_samples[valid_epoch_mask]
            
            events = np.column_stack([
                probe_samples_valid,
                np.zeros(len(probe_samples_valid), dtype=int),
                np.ones(len(probe_samples_valid), dtype=int)
            ])
            
            event_id = {"probe_onset": 1}
            
            epochs = mne.Epochs(
                ecog_reref,
                events=events,
                event_id=event_id,
                tmin=tmin,
                tmax=tmax,
                baseline=None,
                preload=True,
                reject_by_annotation=True
            )
            
            # Plot epoched data interactively
            fig = epochs.plot(
                title=f"{pat_fs} - Probe-onset epochs: -12.1 to -0.1 s",
                scalings=dict(seeg=100e-6, ecog=100e-6),
                n_channels=64,
                show_scrollbars=True,
                n_epochs = 1,
                block=True
            )
                        
            epochs_filename = referenced_filename.replace(".fif", "_probe_pre_onset-epoch.fif")
            epochs.save(epochs_filename, overwrite=True)
            
            print(f"Probe-locked epochs saved to: {epochs_filename}")
            print(f"Number of probe epochs saved: {len(epochs)}")
            
            logger.log_fif_save(
                epochs_filename,
                file_description="Probe-onset locked epochs, -12.1 to -0.1 s"
            )
            
                
#%%
            
            # Finish logging
            logger.finish_log()
            
            if single_file_mode:
                print(f"\n{'='*60}")
                print(f"✅ SINGLE FILE PROCESSING COMPLETE")
                print(f"Patient: {pat_fs}")
                print(f"Movie: {mov}")
                print(f"Log file: {logger.log_file}")
                print(f"{'='*60}\n")
                # Exit after processing single file
                sys.exit(0)
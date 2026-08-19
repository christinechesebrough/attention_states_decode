#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 11 09:50:37 2026

@author: christine
"""

import os, re, sys
from scipy.stats import pearsonr
from itertools import compress
import numpy as np
import pandas as pd
from scipy import stats, signal, interpolate
import matplotlib.pyplot as plt
from scipy.stats import trim_mean


# Define directories

machine_path ='media/christine'#'Volumes' #'media/christine'

fs_dir = f'/{machine_path}/Samsung/anatomy'
#prep_dir = f'/{machine_path}/Samsung/AV40_data/new_converted'
prep_dir = f'/{machine_path}/Samsung/Movie_data/exp_samp_prep_standard'
elec_recon_dir = f'/{machine_path}/Samsung/Movie_data/data/movie_elec_corr_sheets'

frame_dir = f'/{machine_path}/Samsung/Movie_data/data/video_frames'
lum_dir = f'/{machine_path}/Samsung/Movie_data/data/luminance'
exclude_dat = '/media/christine/Samsung/Movie_data/movies_es_power'


freq_bands = ['theta','alpha','beta','gamma','HFA']

yeo7_mapping = {
    "7Networks_1": "Visual",
    "7Networks_2": "Somatomotor",
    "7Networks_3": "Dorsal Attention",
    "7Networks_4": "Ventral Attention",
    "7Networks_5": "Limbic",
    "7Networks_6": "Frontoparietal",
    "7Networks_7": "Default",
}

yeo17_mapping = {
    "17Networks_1":  "Visual Central (Visual A)",
    "17Networks_2":  "Visual Peripheral (Visual B)",
    "17Networks_3":  "Somatomotor A",
    "17Networks_4":  "Somatomotor B",
    "17Networks_5":  "Dorsal Attention A",
    "17Networks_6":  "Dorsal Attention B",
    "17Networks_7":  "Salience / Ventral Attention A",
    "17Networks_8":  "Salience / Ventral Attention B",
    "17Networks_9":  "Limbic A",
    "17Networks_10": "Limbic B",
    "17Networks_11": "Control C",
    "17Networks_12": "Control A",
    "17Networks_13": "Control B",
    "17Networks_14": "Temporal Parietal",
    "17Networks_15": "Default C",
    "17Networks_16": "Default A",
    "17Networks_17": "Default B",
}

output = 'power_z'
pow_type = 'log'

vids = ["inscapes",'betta']#, "inscapes"]

attn_labels = pd.read_csv("/media/christine/Samsung/exp_sampling/logs/all_pat_responses_wide_updated.csv")

# All patients that occur in at least one video
patients = ['LH019']#, 'NS219', 'NS221']
patients.sort()

for pat in patients:

    # Collect ALL videos, recordings, and frequency bands
    # for this patient
    patient_power_dfs = []

    # Load electrode metadata once per patient
    excel_files = sorted([
        f for f in os.listdir(elec_recon_dir)
        if pat in f
        and f.endswith('.xlsx')
        and not f.startswith('.')
    ])

    if not excel_files:
        print(
            f"[SKIP] No correspondence .xlsx for "
            f"{pat} found in {elec_recon_dir}"
        )
        continue

    excel_path = max(
        [os.path.join(elec_recon_dir, f) for f in excel_files],
        key=os.path.getmtime
    )

    elecs_subs = pd.read_excel(excel_path)

    required_cols = {"label"}
    col_map = {
        col.lower(): col
        for col in elecs_subs.columns
    }

    elecs_subs.rename(
        columns={
            col_map[req]: req
            for req in required_cols
        },
        inplace=True
    )


    # ---------------------------------------------------------
    # Now process every video available for this patient
    # ---------------------------------------------------------

    for vid in vids:

        # Patients available for each video
        if vid == "betta":
            vid_patients = [
              'LH019'#  'NS217','NS219','NS221'
            ]

        elif vid == "inscapes":
            vid_patients = ["LH019"
         ]

        # This patient does not have this video
        if pat not in vid_patients:
            print(
                f"[SKIP] {pat} has no {vid} recording")
            continue

        if vid == "inscapes":
            keys = ['inscapes']

        elif vid == "betta":
            keys = ['Betta', 'betta']
            
        attn_labels_run = attn_labels[(attn_labels["pat"] == pat) & (attn_labels["movie"] == vid)]
        # -----------------------------------------------------
        # Frequency bands
        # -----------------------------------------------------

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
    
            fig_dir = f'/{machine_path}/Samsung/Movie_data/{output}_{pow_type}_{freq_band}_{vid}_26Mar26'
            fig_patient_dir = os.path.join(fig_dir, pat)
        
            pow_files = sorted([
                f for f in os.listdir(fig_patient_dir)
                if pat in f
                and f.endswith('.npz')
                and not f.startswith('.')
            ])
            
            if pat == 'LH019':
                if vid == "betta":
                    pow_files = sorted([
                        f for f in os.listdir(fig_patient_dir)
                        if pat in f
                        if 'combined' in f
                        and f.endswith('.npz')
                        and not f.startswith('.')
                    ])
            
            for pow_file in pow_files:
        
                pow_path = os.path.join(fig_patient_dir, pow_file)
                
                suffix = f"_{output}_{pow_type}_epoch_dat.npz"
        
                if pow_file.endswith(suffix):
                    entry_id = pow_file[:-len(suffix)]
                else:
                    entry_id = os.path.splitext(pow_file)[0]
                        
                print(entry_id)
            
                print(f"\nProcessing {pat}, {freq_band}: "f"{pow_file}")
                
            
                dat = np.load(pow_path,allow_pickle=True)
            
                power_epochs = dat["power_epochs"]
                epoch_labels = dat["epoch_labels"]
                channel_names = dat["channel_names"]
                

                epoch_trial_numbers = (
                    pd.Series(epoch_labels)
                    .str.extract(r"QSTART_(\d+)$")[0]
                    .astype(int)
                    .to_numpy()
                )
            
                sfreq = dat["sfreq"].item()
                tmin = dat["tmin"].item()
                tmax = dat["tmax"].item()
                freq_bins = dat["freq_bins"]
                
                samples_per_window = int(sfreq)   # 1 second
            
                n_epochs, n_channels, n_samples = power_epochs.shape
                
                expected_samples = 12 * samples_per_window
                
                if n_samples != expected_samples:
                    raise ValueError(f"Expected {expected_samples} samples for 12 seconds,but power_epochs has {n_samples}")
                
                power_windows = power_epochs.reshape(
                    n_epochs,
                    n_channels,
                    12,
                    samples_per_window
                )
                
                print(f"(shape of {vid} {entry_id}: {power_windows.shape})")
                
                # Robust mean across the 600 samples within each 1-second window
                # Trims lowest 10% and highest 10% before calculating the mean
                power_windows_robust_mean = trim_mean(
                    power_windows,
                    proportiontocut=0.10,
                    axis=3)
                
                print(power_windows_robust_mean.shape)
                print(f"(shape of {vid} {entry_id}: {power_windows_robust_mean.shape})")
                
                # -------------------------------------------------------------
                # Convert epoch x channel x window array to long dataframe
                # -------------------------------------------------------------
                
                n_epochs, n_channels, n_windows = power_windows_robust_mean.shape
                window_times = tmin + np.arange(n_windows)
                power_df = pd.DataFrame({"trial": np.repeat(epoch_labels, n_channels * n_windows), "channel": np.tile(np.repeat(channel_names, n_windows), n_epochs), "time": np.tile(window_times, n_epochs * n_channels), "power": power_windows_robust_mean.reshape(-1)})
                power_df.insert(0, "subject", pat)
                power_df.insert(1, "recording", vid)

                power_df["freq"] = freq_band
                
                power_df["pow_contact_freq"] = (
                    power_df["channel"].astype(str)
                    + "_"
                    + power_df["freq"].astype(str)
                )
                
                power_df["contact_ID"] = (
                    power_df["subject"].astype(str)
                    + "_"
                    + power_df["channel"].astype(str)
                )
                
                power_df["trial"] = (
                    power_df["trial"]
                    .str.extract(r"QSTART_(\d+)$")[0]
                    .astype(int)
                )
                
                attn_labels_run = attn_labels_run.copy()
                attn_labels_run["trial"] = attn_labels_run["trial"].astype(int)
                
                power_df = power_df.merge(
                    attn_labels_run[
                        ["trial", "int_ext", "valence", "physio"]
                    ],
                    on="trial",
                    how="left",
                    validate="many_to_one"
                )  
                
                power_df['attn_label']= power_df['int_ext']
                
                power_df["time"] = power_df.groupby(["trial", "channel"]).cumcount()

                
                exclude_pat_dir = os.path.join(exclude_dat, pat)
                
                exclude_files = os.listdir(exclude_pat_dir)
                
                exclude_matches = [
                    f for f in exclude_files
                    if vid.lower() in f.lower()
                    and f.endswith("window_qc.csv")
                ]
                
                if len(exclude_matches) == 0:
                    raise FileNotFoundError(
                        f"No window QC file found for {pat}, {vid}"
                    )
                
                if len(exclude_matches) > 1:
                    raise ValueError(
                        f"Multiple window QC files found for {pat}, {vid}: {exclude_matches}"
                    )
                
                exclude_file = exclude_matches[0]
                exclude_path = os.path.join(exclude_pat_dir, exclude_file)
                
                exclude_df = pd.read_csv(exclude_path)
                
                print(f"Loaded exclusion file: {exclude_path}")
                
                
                exclude_df["trial"] = exclude_df["parent_epoch"].map(
                    dict(enumerate(epoch_trial_numbers))
                )
                
                exclude_df["trial"] = exclude_df["parent_epoch"].map(
                    dict(enumerate(epoch_trial_numbers))
                )
                
                if exclude_df["trial"].isna().any():
                    raise ValueError(
                        f"Some QC parent_epoch values could not be mapped "
                        f"to retained trials for {pat}, {vid}"
                    )

                
                # Keep only windows marked bad
                bad_windows = (
                    exclude_df.loc[
                        exclude_df["bad"] == True,
                        ["trial", "window"]
                    ]
                    .drop_duplicates()
                    .copy()
                )
                
                # Initialize all power rows as good
                power_df["exclude"] = 0
                
                # Mark matching trial x window rows as excluded
                bad_pairs = pd.MultiIndex.from_frame(
                    bad_windows[["trial", "window"]]
                )
                
                power_pairs = pd.MultiIndex.from_frame(
                    power_df[["trial", "time"]].rename(
                        columns={"time": "window"}
                    )
                )
                
                power_df.loc[
                    power_pairs.isin(bad_pairs),
                    "exclude"
                ] = 1
                
                # -------------------------------------------------------------
                # Create channel metadata from electrode correspondence sheet
                # -------------------------------------------------------------
                
                channel_meta = elecs_subs.copy()
                
                # Standardize channel names before merging
                channel_meta["label"] = (
                    channel_meta["label"]
                    .astype(str)
                    .str.strip()
                )
                
                power_df["channel"] = (
                    power_df["channel"]
                    .astype(str)
                    .str.strip()
                )
                
                # Only rename label -> channel
                # Leave all atlas columns exactly as they already exist
                channel_meta = channel_meta.rename(
                    columns={"label": "channel"}
                )
                
                # Metadata columns to retain
                required_meta_cols = [
                    "channel",
                    "hem",
                    "sEEG_ECoG",
                    "Desikan_Killiany",
                    "Destrieux",
                    "aparc_aseg",
                    "Yeo7",
                    "Yeo17",'lepto_coords_1', 'lepto_coords_2', 'lepto_coords_3',
                    'PIALVOX_coords_1', 'PIALVOX_coords_2', 'PIALVOX_coords_3',
                    'fsaverage_coords_1', 'fsaverage_coords_2', 'fsaverage_coords_3'
                ]
                
            
                
                # Create any genuinely missing columns as NaN
                for col in required_meta_cols:
                    if col not in channel_meta.columns:
                        channel_meta[col] = np.nan
                
                # Keep one metadata row per channel
                channel_meta = channel_meta[required_meta_cols].drop_duplicates(subset="channel")
                
                # Merge channel metadata onto power dataframe
                power_df = power_df.merge(channel_meta, on="channel", how="left")

                final_columns = [
                    "subject",
                    "recording",
                    "trial",
                    "channel",
                    "freq",
                    "pow_contact_freq",
                    "time",
                    "power",
                    'contact_ID',
                    "attn_label",
                    "valence",
                    "physio",
                    "hem",
                  #  "exclude",
                    "sEEG_ECoG",
                    "Desikan_Killiany",
                    "Destrieux",
                    "aparc_aseg",
                    "Yeo7",
                    "Yeo17",'lepto_coords_1', 'lepto_coords_2', 'lepto_coords_3',
                    'PIALVOX_coords_1', 'PIALVOX_coords_2', 'PIALVOX_coords_3',
                    'fsaverage_coords_1', 'fsaverage_coords_2', 'fsaverage_coords_3'
                ]
                
                
                power_df = power_df[final_columns]
                
                patient_power_dfs.append(power_df)
              

    # ---------------------------------------------------------
    # All videos and frequency bands are now complete
    # for this patient
    # ---------------------------------------------------------

    if len(patient_power_dfs) == 0:
        print(
            f"[SKIP] No power data found for {pat}"
        )
        continue

    final_power_df = pd.concat(
        patient_power_dfs,
        ignore_index=True
    )

    final_power_df["Yeo7"] = final_power_df["Yeo7"].replace(yeo7_mapping)
    final_power_df["Yeo17"] =final_power_df["Yeo17"].replace(yeo17_mapping) 

    print(
        f"\nFinal dataframe for {pat}: "
        f"{final_power_df.shape}"
    )

    print(
        final_power_df.groupby(
            ["recording", "freq"]
        ).size()
    )
    
    output_dir = ( f'/{machine_path}/Samsung/Movie_data/final_power_dataframes')
    
    os.makedirs(output_dir,exist_ok=True)
    
    output_path = os.path.join(output_dir,f"{pat}_final_power_dataframe.csv")
    
    final_power_df.to_csv(output_path,index=False)
    
    print(f"Saved: {output_path}")

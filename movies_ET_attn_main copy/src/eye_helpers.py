# -*- coding: utf-8 -*-
"""
Created by Max Tue May 30 14:57:06 2023

The scripting functions to aid in eye tracking preprocessing formerly known as helpers.py

"""

import copy, math
import os
import numpy as np
import pandas as pd
import scipy.interpolate as interp
import scipy.signal as signal
import scipy.stats as stats
from pynwb import NWBHDF5IO
import matplotlib.pyplot as plt
import remodnav
from remodnav.clf import filter_spikes, get_dilated_nan_mask, savgol_filter, median_filter


def combine_left_right(pos_left, pos_right, bad_left, bad_right, t_diff, fs, 
                       interp_kind='linear'):
    
    pos_left[bad_left] = np.nan
    pos_right[bad_right] = np.nan
    
    k_diff = int(np.round(fs * t_diff))
    if k_diff % 2 == 0:
        k_diff += 1      # Odd numbers for centered kernel
    
    pos = np.mean(np.stack((pos_right, pos_left)), axis=0)
    
    val_left_only = np.logical_and(np.invert(bad_left), bad_right)
    val_right_only = np.logical_and(np.invert(bad_right), bad_left)
    
    # Vector of samples
    eye_samples = np.arange(0, len(pos_left))
    
    # Compute the sliding window average difference between left and right
    # gaze position to correct the combined gaze signal
    pos_diff = pos_left - pos_right
    pos_diff_mean = np.convolve(pos_diff, np.ones(k_diff)/k_diff, mode='same')
    
    idx_pos_diff = np.isnan(pos_diff_mean)
    f_pos_diff = interp.interp1d(eye_samples[np.invert(idx_pos_diff)], 
                               pos_diff_mean[np.invert(idx_pos_diff)], 
                               kind='linear', fill_value="extrapolate")
    pos_diff_mean[idx_pos_diff] = f_pos_diff(eye_samples[idx_pos_diff])
    
    pos[val_left_only] = pos_left[val_left_only] - \
        (pos_diff_mean[val_left_only] / 2)
    pos[val_right_only] = pos_right[val_right_only] + \
        (pos_diff_mean[val_right_only] / 2)
        
    # Fill up bad samples
    idx_bad = np.isnan(pos)    
    f = interp.interp1d(eye_samples[idx_bad == 0], pos[idx_bad == 0], 
                        kind=interp_kind, fill_value='extrapolate') 
    pos[idx_bad != 0] = f(eye_samples[idx_bad != 0])
    
    # Fill samples in beginning and end
    idx_diff = np.diff(idx_bad.astype(float))
    
    idx_first = np.where(idx_diff != 0)[0][0]
    idx_last = np.where(idx_diff != 0)[0][-1]
    
    if idx_bad[0] and idx_diff[idx_first] == -1:
        pos[:idx_first+1] = pos[idx_first+1]
        
    if idx_bad[-1] and idx_diff[idx_last] == 1:
        pos[idx_last+1:] = pos[idx_last]
        
    return pos, idx_bad


def interp_bad_samples(data, idx_bad, t, t_buffer_pre, t_buffer_post, fs, 
                       interp_kind='linear'):
    
    # With delays for pupil data
    
    s_buffer_pre = int(np.round(fs * t_buffer_pre))
    s_buffer_post = int(np.round(fs * t_buffer_post))
    
    k_buffer = np.hstack((np.zeros(np.max([0, (s_buffer_post - s_buffer_pre)])), 
                          np.ones(s_buffer_post + s_buffer_pre),
                          np.zeros(np.max([0, (s_buffer_pre - s_buffer_post)]))))
    
    idx_bad = np.convolve(idx_bad, k_buffer, 'same') > 0
    
    f = interp.interp1d(t[idx_bad == 0], data[idx_bad == 0], 
                        kind=interp_kind, fill_value='extrapolate')
    
    data[idx_bad != 0] = f(t[idx_bad != 0])
    
    return data, idx_bad


def interp_bad_data(data_gaze, data_val, k_small, k_large, k_delay, visualize):
   
    """
    Interpolate bad samples in gaze data
    
    Inputs:
        data_gaze       - data array [samples x dimension (xy)]
        data_val        - index of samples with invalid data [samples]
        k_small         - number of samples of small gaps that are filled
        k_large         - number of samples to add to large gaps (blinks, etc)
        k_delay         - number of samples to shift delay around large gaps
        visualize       - flag to plot gaze data before and after processing
        
    Returns:
        data_gaze       - processed data vector
        data_val        - upadted vector of invalid data
    """
    
    data_val = copy.copy(data_val)
    
    # Plot signal
    if visualize:
        plt.figure()
        plt.plot(data_gaze)
    
    # Create sample vector
    samples = np.arange(0, len(data_gaze))
    
    # Interpolate small gaps
    idx_gap = signal.convolve(data_val, np.ones(k_small), 'same')
    gap_val = signal.convolve(idx_gap==0, np.ones((k_small)), mode='same') == 0
    
    gaps = np.logical_and(gap_val, np.invert(data_val))
    data_val[gaps] = True
    
    f_gaze = interp.interp1d(samples[np.invert(gaps)],
                             data_gaze[np.invert(gaps)], 
                             kind='linear', fill_value="extrapolate")
    data_gaze[gaps] = f_gaze(samples[gaps])
    
    # Remove samples around longer gaps (mostly blinks)
    idx_bad = signal.convolve(np.invert(data_val), 
                              np.concatenate([np.zeros(k_delay),
                                              np.ones(k_large)]), 
                              'same') > 1
    
    data_gaze[idx_bad] = np.nan
    data_val[idx_bad] = False
    
    if visualize:
        plt.plot(data_gaze)
        
    return data_gaze, data_val


def detect_blinks(val_left, val_right, t_nwb, fs_eye, min_duration=0.09, max_duration=0.5):
    """
    Detect blinks based on validity values for left and right eyes.
    
    Inputs:
        val_left       - Validity array for left eye (True = valid, False = invalid).
        val_right      - Validity array for right eye (True = valid, False = invalid).
        t_nwb          - Timestamps for gaze data.
        fs_eye         - Sampling rate of eye-tracking data (Hz).
        min_duration   - Minimum duration of a blink (in seconds).
        max_duration   - Maximum duration of a blink (in seconds).
    
    Returns:
        blink_events   - List of blink events with start and end times and durations.
    """
    # Combine validity arrays to find invalid periods
    invalid_data = ~val_left & ~val_right  # Both eyes invalid
    
    # Identify start and end indices of invalid periods
    invalid_starts = np.where(np.diff(invalid_data.astype(int)) == 1)[0] + 1
    invalid_ends = np.where(np.diff(invalid_data.astype(int)) == -1)[0] + 1

    # Handle case where the data starts or ends with invalid periods
    if invalid_data[0]:
        invalid_starts = np.insert(invalid_starts, 0, 0)
    if invalid_data[-1]:
        invalid_ends = np.append(invalid_ends, len(invalid_data) - 1)

    # Calculate durations of invalid periods
    durations = (invalid_ends - invalid_starts) / fs_eye

    # Classify blinks based on duration thresholds
    blink_events = []
    for start, end, duration in zip(invalid_starts, invalid_ends, durations):
        if min_duration <= duration <= max_duration:
            blink_events.append({
                "start_time": t_nwb[start],
                "end_time": t_nwb[end],
                "duration": duration
            })
    
    return blink_events


def save_blink_events(blink_events, output_dir, subject, movie):
    """
    Save blink events to a CSV file.
    
    Inputs:
        blink_events   - List of blink event dictionaries.
        output_dir     - Directory to save the CSV file.
        subject        - Subject ID for naming the file.
        movie          - Movie name for naming the file.
    """
    # Convert blink events to a DataFrame
    df_blinks = pd.DataFrame(blink_events)

    # Construct file path
    output_file = os.path.join(output_dir, f"{subject}_{movie}_blink_events.csv")

    # Save to CSV
    df_blinks.to_csv(output_file, index=False)
    print(f"Blink events saved to {output_file}")


def calculate_vergence_measures(nwb_data, screen_pix, screen_cm, k_small=15, k_large=90, 
                               k_delay=20, vis_bads=False, interp_kind='linear'):
    """
    Calculate vergence measures from NWB eye tracking data.
    
    Inputs:
        nwb_data       - NWB object containing eye tracking data
        screen_pix     - Screen dimensions in pixels [width, height]
        screen_cm      - Screen dimensions in cm [width, height]
        k_small        - Number of samples for small gap interpolation
        k_large        - Number of samples for large gap interpolation
        k_delay        - Number of samples to shift delay around large gaps
        vis_bads       - Flag to visualize bad data processing
        interp_kind    - Interpolation method ('linear', 'cubic', etc.)
    
    Returns:
        vergence_data  - Dictionary containing all vergence measures
    """
    # Extract gaze data
    l_gaze = nwb_data.processing['eye_tracking']['eyes']['l_eye_adcs'].data[:]
    r_gaze = nwb_data.processing['eye_tracking']['eyes']['r_eye_adcs'].data[:]
    
    x_right = r_gaze[:,0]
    y_right = r_gaze[:,1]
    x_left = l_gaze[:,0]
    y_left = l_gaze[:,1]
    
    # Extract eye position data
    eye_pos_left = nwb_data.processing['eye_tracking']['eyes']['l_eye_pos'].data[:]
    eye_pos_right = nwb_data.processing['eye_tracking']['eyes']['r_eye_pos'].data[:]
    gaze_pos_left = nwb_data.processing['eye_tracking']['eyes']['l_eye_gaze'].data[:]
    gaze_pos_right = nwb_data.processing['eye_tracking']['eyes']['r_eye_gaze'].data[:]
    
    # Calculate interpupillary distance
    distances = np.sqrt(np.sum((eye_pos_left - eye_pos_right) ** 2, axis=1))
    IPD_cm = np.median(distances) / 10  # Convert from mm to cm
    
    # Extract timestamps and validity
    t_nwb = nwb_data.processing['eye_tracking']['eyes']['l_eye_adcs'].timestamps[:]
    val_left = nwb_data.processing['eye_tracking']['eyes']['l_eye_adcs'].control[:] <= 1
    val_right = nwb_data.processing['eye_tracking']['eyes']['r_eye_adcs'].control[:] <= 1
    
    # Calculate sampling rate
    fs_eye = 1/np.median(np.diff(t_nwb))
    
    # Detect blinks
    blink_events = detect_blinks(val_left, val_right, t_nwb, fs_eye)
    
    # Interpolate bad data for eye positions and distances
    eye_pos_left_interp, _ = interp_bad_data(eye_pos_left[:, 2], val_left, k_small, k_large, k_delay, vis_bads)
    eye_pos_right_interp, _ = interp_bad_data(eye_pos_right[:, 2], val_left, k_small, k_large, k_delay, vis_bads)
    gaze_pos_left_interp, _ = interp_bad_data(gaze_pos_left[:, 2], val_left, k_small, k_large, k_delay, vis_bads)
    gaze_pos_right_interp, _ = interp_bad_data(gaze_pos_right[:, 2], val_left, k_small, k_large, k_delay, vis_bads)
    
    # Calculate distances
    dist_left = eye_pos_left_interp - gaze_pos_left_interp
    dist_right = eye_pos_right_interp - gaze_pos_right_interp
    
    # Convert to cm
    dist_left_cm = dist_left / 10
    dist_right_cm = dist_right / 10
    avg_dist = (dist_left_cm + dist_right_cm) / 2
    
    # Convert gaze positions to pixel dimensions
    x_left = x_left * screen_pix[0]
    x_right = x_right * screen_pix[0]
    
    # Create data structures
    data_left = np.core.records.fromarrays([x_left], names=['x'])
    data_right = np.core.records.fromarrays([x_right], names=['x'])
    
    # Calculate pixel to degree conversion
    px2deg_x_left = np.rad2deg(2 * np.arctan(screen_cm[0] / (2 * dist_left_cm))) / screen_pix[0]
    px2deg_x_right = np.rad2deg(2 * np.arctan(screen_cm[0] / (2 * dist_right_cm))) / screen_pix[0]
    
    # Apply DVA conversion
    data_left['x'] *= px2deg_x_left
    data_right['x'] *= px2deg_x_right
    
    # Calculate gaze disparity
    gaze_disparity_x = data_right['x'] - data_left['x']
    
    # Interpolate gaze disparity
    valid_disp_x = ~np.isnan(gaze_disparity_x)
    eye_samples_x = np.arange(len(gaze_disparity_x))
    
    f_x_disp = interp.interp1d(eye_samples_x[valid_disp_x], gaze_disparity_x[valid_disp_x], 
                               kind=interp_kind, fill_value='extrapolate')
    x_disp_filled = f_x_disp(eye_samples_x)
    
    # Cap interpolated values
    std_gaze_disp_x = np.nanstd(gaze_disparity_x)
    cap_value = 2 * std_gaze_disp_x
    interp_idx = np.isnan(gaze_disparity_x)
    x_disp_filled_capped = x_disp_filled.copy()
    x_disp_filled_capped[interp_idx] = np.clip(x_disp_filled[interp_idx], -cap_value, cap_value)
    
    # Calculate visual focus displacement (Huang et al. 2019)
    beta = 0.283
    PD_cm = IPD_cm
    
    visual_focus_displacement = np.zeros(len(gaze_disparity_x))
    
    for i in range(len(gaze_disparity_x)):
        G = np.linalg.norm(gaze_disparity_x[i])
        E = beta * G
        
        if gaze_disparity_x[i] > 0:  # Divergence
            visual_focus_displacement[i] = E * (dist_left_cm[i] + dist_right_cm[i]) * 5 / (PD_cm * 10 - E)
        else:  # Convergence
            visual_focus_displacement[i] = -E * (dist_left_cm[i] + dist_right_cm[i]) * 5 / (PD_cm * 10 + E)
    
    # Interpolate visual focus displacement
    valid_vis_fd = ~np.isnan(visual_focus_displacement)
    eye_samples_fd = np.arange(len(visual_focus_displacement))
    
    f_vis_fd = interp.interp1d(eye_samples_fd[valid_vis_fd], visual_focus_displacement[valid_vis_fd], 
                               kind=interp_kind, fill_value='extrapolate')
    vis_fd_filled = f_vis_fd(eye_samples_fd)
    
    # Cap visual focus displacement
    std_vis_fd = np.nanstd(visual_focus_displacement)
    cap_value_vf = 3 * std_vis_fd
    interpolated_indices = np.isnan(visual_focus_displacement)
    vis_fd_filled_capped = vis_fd_filled.copy()
    vis_fd_filled_capped[interpolated_indices] = np.clip(vis_fd_filled[interpolated_indices], -cap_value_vf, cap_value_vf)
    
    # Compile results
    vergence_data = {
        'time': t_nwb,
        'gaze_dist_x_raw': x_right - x_left,
        'gaze_dist_x_interp': x_disp_filled,
        'dva_gaze_disp_x': gaze_disparity_x,
        'dva_gaze_disp_x_interp': x_disp_filled_capped,
        'vis_focus_disp': visual_focus_displacement,
        'vis_fd_interp': vis_fd_filled_capped,
        'blink_events': blink_events,
        'IPD_cm': IPD_cm,
        'fs_eye': fs_eye
    }
    
    return vergence_data


def interp_gaps(gaze):
    
    eye_samples = np.arange(0, len(gaze))
    
    idx_gap = np.isnan(gaze)

    f_gaps = interp.interp1d(eye_samples[np.invert(idx_gap)], gaze[np.invert(idx_gap)], 
                             kind='linear', fill_value="extrapolate")
    
    gaze[idx_gap] = f_gaps(eye_samples[idx_gap])
    
    return gaze, idx_gap


def prep_gaze(xy_right, xy_left, gaze_val, t, fs):
    
    """
    Load and preprocess gaze data
    
    Inputs:
        data_file       - full path to eyetracking data file in csv format
        et_prep_params  - object defining preprocessing parameters
        
    Returns:
        xy              - gaze data in xy coordinated [DVA] [samples x dimension]
        fs_eye          - upadted vector of invalid data
        t               - array of timestamps
        screen_dva      - screen dimension in DVA
    """
         
    k_art_fill=0.025
    k_art_buffer=0.25
    k_art_delay=0.033
    vis_art_prep=False
    xy_mean_k=0.1
    med_filt=0.04
    screen_dist=60
    screen_diag=23.8*2.54
    screen_ar=16/9
    
    # Get position and time
    x_right = xy_right[:, 0]
    y_right = xy_right[:, 1]
    
    x_left = xy_left[:, 0]
    y_left = xy_left[:, 1]
    
    val_left = gaze_val[:, 0]
    val_right = gaze_val[:, 1]
    
    #%% Preprocess eyetracking data
    
    # Exclude data outside the screen    
    val_left[np.logical_or(x_left > 1, x_left < 0)] = False
    val_left[np.logical_or(y_left > 1, y_left < 0)] = False
    
    val_right[np.logical_or(x_right > 1, x_right < 0)] = False
    val_right[np.logical_or(y_right > 1, y_right < 0)] = False
    
    x_left[np.invert(val_left)] = np.nan
    y_left[np.invert(val_left)] = np.nan
    
    x_right[np.invert(val_right)] = np.nan
    y_right[np.invert(val_right)] = np.nan
    
    # Interpolate data                 
    k_art_fill = int(round(k_art_fill * fs))
    k_art_buffer = int(round(k_art_buffer * fs))
    k_art_delay = int(round(k_art_delay * fs))
    
    x_left, _ = interp_bad_data(x_left, 
                                val_left, 
                                k_art_fill, 
                                k_art_buffer, 
                                k_art_delay, 
                                vis_art_prep)
    y_left, val_left = interp_bad_data(y_left, 
                                       val_left, 
                                       k_art_fill, 
                                       k_art_buffer, 
                                       k_art_delay, 
                                       vis_art_prep)
    
    x_right, _ = interp_bad_data(x_right, 
                                 val_right, 
                                 k_art_fill, 
                                 k_art_buffer, 
                                 k_art_delay, 
                                 vis_art_prep)
    y_right, val_right = interp_bad_data(y_right, 
                                         val_right, 
                                         k_art_fill, 
                                         k_art_buffer, 
                                         k_art_delay, 
                                         vis_art_prep)
    
    # Average left and right eye
    x, bad_samples_x = combine_left_right(x_left, x_right, np.invert(val_left), np.invert(val_right), 
                                          xy_mean_k, fs)
    y, bad_samples_y = combine_left_right(y_left, y_right, np.invert(val_left), np.invert(val_right), 
                                          xy_mean_k, fs)
    
    # Concatenate data
    xy = np.concatenate((np.expand_dims(x,1), np.expand_dims(y,1)), axis=1)
    
    # Median filter
    k_med_filt = int(round(med_filt * fs))
    if k_med_filt % 2 == 0:
        k_med_filt += 1      # Odd numbers for centered kernel
        
    xy = signal.medfilt(xy, (k_med_filt, 1))
    
    # Remove bad samples
    xy[bad_samples_x, 0] = np.nan
    xy[bad_samples_y, 1] = np.nan
    
    #%% Convert eye position to dva
    dist = screen_dist
    
    # Screen dimentions [cm]
    d = screen_diag
    ar = screen_ar
    
    # Compute width and height
    h = d * math.sin(math.atan(1/ar))
    w = d * math.cos(math.atan(1/ar))
    
    # Scale to cm
    xy = (xy - 0.5) * np.array([w,h])
    
    # Convert to DVA
    xy = np.rad2deg(2 * np.arctan(xy / (2*dist)))
    
    # Maybe flip y axis?
    xy[:,1] = -xy[:,1]
    
    # Screen
    screen_dva = np.rad2deg(2 * np.arctan(np.array([w, h]) / (2*dist)))
    
    #%% Return variables
    return xy, screen_dva

#def detect_saccades_remodnav(nwb_fname, k_small, k_large, k_delay, vis_bads, t_diff, screen_cm, screen_pix, savgol_length):
def detect_saccades_remodnav(nwb_fname, k_small, k_large, k_delay, vis_bads, t_diff, screen_cm, screen_pix, savgol_length):

    
    #%% Load the data from nwb files
    nwb_io = NWBHDF5IO(nwb_fname, mode='r', load_namespaces=True)
    nwb = nwb_io.read()
    
    # Get position and time
    l_gaze = nwb.processing['eye_tracking']['eyes']['l_eye_adcs'].data[:]
    r_gaze = nwb.processing['eye_tracking']['eyes']['r_eye_adcs'].data[:]

    x_right = r_gaze[:,0]
    y_right = r_gaze[:,1]
    
    x_left = l_gaze[:,0]
    y_left = l_gaze[:,1]
    
    # replace nans with 0s for consistency with old version
   # x_left[np.isnan(x_left)] = 0
   # y_left[np.isnan(y_left)] = 0
   # x_right[np.isnan(x_right)] = 0
   # y_right[np.isnan(y_right)] = 0
    
    # Distance to screen 
    eye_pos_left = nwb.processing['eye_tracking']['eyes']['l_eye_pos'].data[:]
    eye_pos_right = nwb.processing['eye_tracking']['eyes']['r_eye_pos'].data[:]
    
    gaze_pos_left = nwb.processing['eye_tracking']['eyes']['l_eye_gaze'].data[:]
    gaze_pos_right = nwb.processing['eye_tracking']['eyes']['r_eye_gaze'].data[:]
    

    # Replace NaNs in eye position arrays with 0
   # eye_pos_left[np.isnan(eye_pos_left)] = 0
   # eye_pos_right[np.isnan(eye_pos_right)] = 0
    
    # Replace NaNs in gaze position arrays with 0
   # gaze_pos_left[np.isnan(gaze_pos_left)] = 0
   # gaze_pos_right[np.isnan(gaze_pos_right)] = 0
    
        
    # Validity
    val_left = nwb.processing['eye_tracking']['eyes']['l_eye_adcs'].control[:] <= 1
    val_right = nwb.processing['eye_tracking']['eyes']['r_eye_adcs'].control[:] <= 1
    
    t_nwb = nwb.processing['eye_tracking']['eyes']['l_eye_adcs'].timestamps[:]
    
    # Sampling rate
    fs_eye = 1/np.median(np.diff(t_nwb))
    
    #%% Preprocess eyetracking data
    
    # Exclude data outside the screen    
    
    
    val_left[np.logical_or(x_left > 1, x_left < 0)] = False
    val_left[np.logical_or(y_left > 1, y_left < 0)] = False
    
    val_right[np.logical_or(x_right > 1, x_right < 0)] = False
    val_right[np.logical_or(y_right > 1, y_right < 0)] = False
    
    prop_valid_left = np.sum(val_left) / len(val_left)
    prop_valid_right = np.sum(val_right) / len(val_right)

    print(f"Proportion valid (left eye): {prop_valid_left:.3f}")
    print(f"Proportion valid (right eye): {prop_valid_right:.3f}")
    
    # Interpolate data
    x_left, _ = interp_bad_data(x_left, val_left, k_small, k_large, k_delay, vis_bads)
    y_left, val_left = interp_bad_data(y_left, val_left, k_small, k_large, k_delay, vis_bads)
    
    x_right, _ = interp_bad_data(x_right, val_right, k_small, k_large, k_delay, vis_bads)
    y_right, val_right = interp_bad_data(y_right, val_right, k_small, k_large, k_delay, vis_bads)
    
    eye_pos_left, _ = interp_bad_data(eye_pos_left[:,2], val_left, k_small, k_large, k_delay, vis_bads)
    eye_pos_right, _ = interp_bad_data(eye_pos_right[:,2], val_left, k_small, k_large, k_delay, vis_bads)
    gaze_pos_left, _ = interp_bad_data(gaze_pos_left[:,2], val_left, k_small, k_large, k_delay, vis_bads)
    gaze_pos_right, _ = interp_bad_data(gaze_pos_right[:,2], val_left, k_small, k_large, k_delay, vis_bads)
    
    dist_left = np.nanmedian(eye_pos_left - gaze_pos_left)
    dist_right = np.nanmedian(eye_pos_right - gaze_pos_right)
    
    # Distance in cm
    dist = np.mean([dist_left, dist_right]) / 10
    
    # If only data from one eye is good, use that one
    x, bad_samples_x = combine_left_right(x_left, x_right, np.invert(val_left), np.invert(val_right), t_diff, fs_eye)
    y, bad_samples_y = combine_left_right(y_left, y_right, np.invert(val_left), np.invert(val_right), t_diff, fs_eye)
    
    # Concatenate data
    xy = np.concatenate((np.expand_dims(x,1), np.expand_dims(y,1)), axis=1)
    
    xy[bad_samples_x, 0] = np.nan
    xy[bad_samples_y, 1] = np.nan
    
    #%% Convert eye position to dva

    # Screen
    screen_dva = np.rad2deg(2 * np.arctan(screen_cm / (2*dist)))

    # Screen dimenstions
    px2deg = np.mean(screen_dva / screen_pix)
    
    xy = xy * screen_pix
    
    # Setup data
    n = len(xy)
    
    data = np.core.records.fromarrays([
            xy[:,0],
            xy[:,1],
            [0.0] * n,
            [0] * n],
            names=['x', 'y', 'pupil', 'frame'])
    
    # Defaults
    saccades_class = remodnav.EyegazeClassifier(px2deg, fs_eye,
                                                noise_factor=5,
                                                min_saccade_duration=0.01,
                                                min_fixation_duration=0.04,
                                                max_initial_saccade_freq=2,
                                                saccade_context_window_length=1)
    
    #eye_prep = saccades_class.preproc(data,dilate_nan=0.2, median_filter_length=0.05, savgol_length = savgol_length,savgol_polyord=2)
    
    eye_prep, warning_indices = preproc_with_warnings(saccades_class, data,dilate_nan=0.2, median_filter_length=0.05, savgol_length=savgol_length, savgol_polyord=2)

    
    events = saccades_class(eye_prep)
    events = pd.DataFrame(events)
    
    t_eye = np.arange(0, len(xy)) / fs_eye
    
    sacc_label = [np.in1d(l, ['SACC', 'ISAC'])[0] 
                  for l in events.label]
    
    saccade_remodnav = events[sacc_label].start_time.values
    fixation_remodnav = events[sacc_label].end_time.values
    
    start_pos = np.concatenate((np.expand_dims(events[sacc_label].start_x.values, axis=1), 
                                np.expand_dims(events[sacc_label].start_y.values, axis=1)), 
                               axis=1)
    
    end_pos = np.concatenate((np.expand_dims(events[sacc_label].end_x.values, axis=1), 
                              np.expand_dims(events[sacc_label].end_y.values, axis=1)), 
                             axis=1)
    
    saccade_pos = np.concatenate((start_pos, end_pos), axis=1)
    
    # Calculate saccade amplitudes and velocities
    
    # Pixel difference
    dx = events[sacc_label].end_x.values - events[sacc_label].start_x.values
    dy = events[sacc_label].end_y.values - events[sacc_label].start_y.values
    dist_px = np.sqrt(dx**2 + dy**2)
    
    # Convert to degrees using px2deg
    dist_deg = dist_px * px2deg
    
    # Duration (seconds)
    dur = events[sacc_label].end_time.values - events[sacc_label].start_time.values
    
    # Velocity (deg/s)
    velocity = dist_deg / dur
    
    # Identify implausible saccades
    implausible = velocity > 1000
    
    # Count and report
    num_implausible = np.sum(implausible)
    max_velocity = np.max(velocity) if len(velocity) > 0 else np.nan
    
    print(f"Number of implausible saccades (>1000 deg/s): {num_implausible}")
    print(f"Maximum saccade velocity detected: {max_velocity:.1f} deg/s")
    
        
    # Timing     
    f_time = interp.interp1d(t_eye, np.arange(0, len(t_eye)))

    saccade_onset_vec = np.zeros(t_eye.shape)
    saccade_onset_vec[np.round(f_time(saccade_remodnav)).astype(int)] = 1
    
    fixation_vec = np.zeros(t_eye.shape)
    fixation_vec[np.round(f_time(fixation_remodnav)).astype(int)] = 1
    
    #return saccade_onset_vec, fixation_vec, saccade_pos, xy, t_eye, t_nwb, fs_eye, num_implausible, max_velocity
    return saccade_onset_vec, fixation_vec, saccade_pos, xy, t_eye, t_nwb, fs_eye, num_implausible, max_velocity, warning_indices

#%%

def detect_saccades(xy, t, fs_eye):
    """
    Detect Saccades
    
    Inputs:
        xy                  - Gaze data in DVA on x and y coordinates [samples x dimensions]
        t                   - timestamps [samples]
        fs_eye              - Sampling rate of gaze data [Hz]
        condition           - Condition to analyze data [string: 'auditory', 'visual', 'vsearch']
        trial_et_time       - Time of trial start [s]
        response_et_time    - End of each trial [s]
        trial_condition     - Experimental condition for each trial
        et_prep_params      - object defining parameters
        
    Returns:
        saccade_onset_t     - Time of saccade onset
        fixation_onset_t    - Time of fixation onset
        saccade_condition   - Experiment condition of saccade 
    """
    sacc_kernel=0.00667
    med_filt_v=0.01
    med_filt_noise=0.1667
    v_med_thresh=5
    k_noise_min=0.1
    k_noise_dil=0.1667
    k_buffer_blink=0.002
    sacc_thresh=5
    sacc_min_dist=0.05
    sacc_win=0.05
    sacc_on_thresh=4
    
    #%% 'Edges' in gaze data
    # Convert kernel to samples 
    k_size = int(round(sacc_kernel * fs_eye))
    if k_size % 2 == 0:
        k_size += 1      # Odd numbers for centered kernel
    
    kernel_diff = np.hstack((np.ones(k_size), 0, -1*np.ones(k_size)))
    x_diff = np.convolve(xy[:,0], kernel_diff, 'same')
    y_diff = np.convolve(xy[:,1], kernel_diff, 'same')
    
    gaze_diff = np.mean(np.vstack((np.abs(x_diff), np.abs(y_diff))), axis=0)
    
    #%% Detect saccades
    # Convert kernel to samples 
    k_med_filt_v = int(round(med_filt_v * fs_eye))
    if k_med_filt_v % 2 == 0:
        k_med_filt_v += 1      # Odd numbers for centered kernel
        
    v = np.sqrt(np.diff(xy[:,0])**2 + np.diff(xy[:,1])**2)
    v = np.hstack((0, v))
    
    v = signal.medfilt(v, k_med_filt_v)
    
    # Fill in bad samples
    idx_nan = np.isnan(v)
    f = interp.interp1d(t[np.invert(idx_nan)], v[np.invert(idx_nan)], 
                        fill_value='extrapolate')
    v[idx_nan] = f(t[idx_nan])
    
    # Compute rolling median to estimate noise
    med_len = int(round(med_filt_noise * fs_eye))
    v_med = pd.Series(v).rolling(med_len).median().values
    v_med = np.hstack((v_med[int(med_len/2):], np.zeros(int(med_len/2))))
    
    v[idx_nan] = np.nan
    v_med[idx_nan] = np.nan
    
    noise_thresh = v_med_thresh * stats.iqr(v_med[np.invert(np.isnan(v_med))])
    idx_noise = v_med > noise_thresh 
    
    # Erode smaller regions 
    # Convert kernel to samples 
    k_noise_min = int(round(k_noise_min * fs_eye))
    if k_noise_min % 2 == 0:
        k_noise_min += 1      # Odd numbers for centered kernel
        
    idx_noise = np.convolve(idx_noise == 0, np.ones(k_noise_min), mode='same') == 0
    idx_noise = np.convolve(idx_noise, np.ones(k_noise_min), mode='same') > 0
    
    # Dilate noisy segments
    k_noise_dil = int(round(k_noise_dil * fs_eye))
    if k_noise_dil % 2 == 0:
        k_noise_dil += 1      # Odd numbers for centered kernel
        
    idx_noise = np.convolve(idx_noise, np.ones(k_noise_dil), mode='same') > 0
    
    # Remove those close to blinks
    k_buffer_blink = int(round(k_buffer_blink * fs_eye))
    
    idx_nan = np.logical_or(np.isnan(xy[:,0]), np.isnan(xy[:,1]))
    idx_nan = signal.convolve(idx_nan, np.ones(k_buffer_blink), 'same') > 0
    
    # Detect saccades 
    sacc_thr = sacc_thresh * stats.iqr(gaze_diff[np.invert(np.isnan(gaze_diff))])
    
    sacc_min_dist = int(round(sacc_min_dist * fs_eye))
    idx_saccade, _ = signal.find_peaks(gaze_diff, 
                                        height=sacc_thr,
                                        distance=sacc_min_dist,
                                        prominence=sacc_thr)
    
    # Remove saccades in noisy segments
    idx_saccade = np.delete(idx_saccade, np.in1d(idx_saccade, 
                                                  np.where(idx_noise)))
    
    # Remove saccades close to blinks
    idx_saccade = np.delete(idx_saccade, np.in1d(idx_saccade, 
                                                  np.where(idx_nan)))
    
    # Find more precise timing of saccade onset and offset
    win_sacc = int(round(sacc_win * fs_eye))
    sacc_thr = sacc_on_thresh * stats.iqr(gaze_diff[np.invert(np.isnan(gaze_diff))])
  
    sacc_onset = np.empty(len(idx_saccade))
    fix_onset = np.empty(len(idx_saccade))
    
    for i, isacc in enumerate(idx_saccade):
    
        idx_start = isacc - win_sacc
        idx_end = isacc + win_sacc
           
        g_diff_win = gaze_diff[idx_start : idx_end] 
    
        idx_sacc_win = g_diff_win > np.nanmedian(gaze_diff) + sacc_thr
        idx_sacc_diff = np.hstack([0, np.diff(idx_sacc_win.astype(int))])
        
        sacc_onset_win = np.where(idx_sacc_diff == 1)[0] + np.round(k_size/2)
        fix_onset_win = np.where(idx_sacc_diff == -1)[0] - np.round(k_size/2)  

        if len(sacc_onset_win) == 0:
            sacc_onset_win = win_sacc
        elif len(sacc_onset_win) > 1:
            sacc_onset_win = sacc_onset_win[np.argmin(np.abs(sacc_onset_win - win_sacc))]
                
        if len(fix_onset_win) == 0:
            fix_onset_win = win_sacc
        elif len(fix_onset_win) > 1:
            fix_onset_win = fix_onset_win[np.argmin(np.abs(fix_onset_win - win_sacc))]

        sacc_onset[i] = int(isacc - (win_sacc - sacc_onset_win))
        fix_onset[i] = int(isacc - (win_sacc - fix_onset_win))
        
    plt.figure()
    plt.plot(xy, label=['x', 'y'])
    plt.plot(10*v, label='v')
    plt.plot(gaze_diff, label='gaze_diff')
    plt.plot(100*v_med, label='median v')
    plt.plot(idx_noise, label='noise')

    y_lim = plt.ylim()
    
    for sac in sacc_onset:        
        plt.plot(sac*np.ones(2), y_lim, 'k--', linewidth=0.5)
        
    plt.title('std = {:1.2f}, Average median velocity = {:1.3f}'.format(
        np.nanstd(xy), np.nanmean(v_med)))
    plt.legend()
    
    # Convert to time
    saccade_onset_t = t[sacc_onset.astype(int)]
    fixation_onset_t = t[fix_onset.astype(int)]
                    
    return saccade_onset_t, fixation_onset_t

#%%


def plot_warning_indices_and_saccades_on_gaze(t_nwb, xy, warning_indices, saccade_onset_vec, title="Warning indices and saccades on gaze X"):
    """
    Plot gaze X position with warning indices and identified saccade onsets overlaid.

    Parameters:
        t_nwb : np.ndarray
            Time vector.
        xy : np.ndarray
            Gaze positions [samples x 2].
        warning_indices : list or np.ndarray
            Indices of warnings to overlay.
        saccade_onset_vec : np.ndarray
            Binary vector indicating saccade onsets (same length as xy).
        title : str
            Plot title.
    """
    plt.figure(figsize=(12,4))
    
    # Plot gaze X position
    plt.plot(t_nwb, xy[:,0], label='Gaze X', color='blue')
    
    # Plot warning indices
    if len(warning_indices) > 0:
        plt.scatter(t_nwb[warning_indices], xy[warning_indices,0], color='red', s=20, label='Warning indices')
    
    # Plot saccade onsets
    saccade_indices = np.where(saccade_onset_vec == 1)[0]
    if len(saccade_indices) > 0:
        plt.scatter(t_nwb[saccade_indices], xy[saccade_indices,0], color='pink', s=20, label='Detected saccades')

    plt.xlabel('Time (s)')
    plt.ylabel('X position (pix)')
    plt.title(title)
    plt.legend()
    plt.show()



#%%
def preproc_with_warnings(saccades_class, data, min_blink_duration=0.02, dilate_nan=0.01,
                          median_filter_length=0.05, savgol_length=0.019, savgol_polyord=2, max_vel=1000.0):
    """
    Wrapper replicating remodnav clf.preproc but returns warning_indices for diagnostics.
    """
    # Convert params in seconds to #samples
    dilate_nan = int(dilate_nan * saccades_class.sr)
    min_blink_duration = int(min_blink_duration * saccades_class.sr)

    # Sanity check savgol_length
    if (int(savgol_length * saccades_class.sr) % 2 != 1 or
        int(savgol_length * saccades_class.sr) < savgol_polyord) and savgol_length != 0.0:
        raise ValueError("Invalid savgol_length window size.")
    savgol_length = int(savgol_length * saccades_class.sr)
    median_filter_length = int(median_filter_length * saccades_class.sr)

    # In-place spike filter
    data = filter_spikes(data)

    # Dilate NaN segments
    if dilate_nan:
        mask = get_dilated_nan_mask(
            data['x'],
            dilate_nan,
            min_blink_duration)
        data['x'][mask] = np.nan
        data['y'][mask] = np.nan

    # Savitzky-Golay smoothing
    if savgol_length:
        for i in ('x', 'y'):
            data[i] = savgol_filter(data[i], savgol_length, savgol_polyord)

    # Velocity calculation
    velocities = saccades_class._get_velocities(data)

    # Median filtered velocities
    if median_filter_length:
        med_velocities = np.zeros((len(data),), velocities.dtype)
        med_velocities[1:] = (
            np.diff(median_filter(data['x'], size=median_filter_length)) ** 2 +
            np.diff(median_filter(data['y'], size=median_filter_length)) ** 2) ** 0.5
        med_velocities *= saccades_class.px2deg * saccades_class.sr
        med_velocities[get_dilated_nan_mask(med_velocities, dilate_nan, 0)] = np.nan
    else:
        med_velocities = velocities

    # Replace "too fast" velocities with previous velocity, record warning indices
    filtered_velocities = [float(0)]
    warning_indices = []
    for i, vel in enumerate(velocities):
        if vel > max_vel:
            print(f"Computed velocity exceeds threshold [{vel:.1f} > {max_vel} deg/s]")
            warning_indices.append(i+1)
            vel = filtered_velocities[-1]
        filtered_velocities.append(vel)
    velocities = np.array(filtered_velocities)

    # Acceleration
    acceleration = np.zeros(velocities.shape, velocities.dtype)
    acceleration[1:] = (velocities[1:] - velocities[:-1]) * saccades_class.sr

    # Create output structured array
    arrs = [med_velocities, velocities, acceleration, data['x'], data['y']]
    names = ['med_vel', 'vel', 'accel', 'x', 'y']

    eye_prep = np.core.records.fromarrays(arrs, names=names)

    return eye_prep, warning_indices


#%%
def warnings_near_saccades(warning_indices, saccade_onset_vec, fs_eye, window_ms=10):
    """
    Check how many warning indices are within +/- window_ms of detected saccade onsets.

    Parameters:
        warning_indices : list or np.ndarray
            Indices where velocity exceeded max_vel.
        saccade_onset_vec : np.ndarray
            Binary vector indicating detected saccade onsets.
        fs_eye : float
            Sampling rate in Hz.
        window_ms : float
            Time window in milliseconds to define 'near'.

    Returns:
        summary : dict
            Counts and percentages of warnings near saccades vs isolated.
    """
    # Convert window to samples
    window_samples = int(np.round((window_ms / 1000) * fs_eye))

    # Get indices of saccade onsets
    saccade_indices = np.where(saccade_onset_vec == 1)[0]

    # Initialize counts
    near_count = 0
    isolated_count = 0

    # Check each warning
    for w_idx in warning_indices:
        # Is it within +/- window_samples of any saccade index?
        if np.any(np.abs(saccade_indices - w_idx) <= window_samples):
            near_count += 1
        else:
            isolated_count += 1

    total_warnings = len(warning_indices)

    summary = {
        'total_warnings': total_warnings,
        'near_saccades': near_count,
        'isolated_warnings': isolated_count,
        'percent_near_saccades': (near_count / total_warnings * 100) if total_warnings > 0 else 0,
        'percent_isolated': (isolated_count / total_warnings * 100) if total_warnings > 0 else 0
    }

    # Print summary
    print(f"Total warnings: {total_warnings}")
    print(f"Warnings near saccades (+/- {window_ms} ms): {near_count} ({summary['percent_near_saccades']:.1f}%)")
    print(f"Isolated warnings: {isolated_count} ({summary['percent_isolated']:.1f}%)")

    return summary

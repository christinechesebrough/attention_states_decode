#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  4 21:03:14 2025

@author: christinechesebrough

Helper functions for EEG preprocessing pipeline.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from mne.time_frequency import psd_array_welch
from mne.filter import filter_data
import mne
from scipy import signal
import re
from collections import defaultdict


def save_bad_channels(raw_data, file_path, patient_id, movie_base):
    """
    Save bad channels to a text file with metadata.
    
    Parameters:
    -----------
    raw_data : mne.io.Raw
        The MNE Raw object containing the data
    file_path : str
        Path where to save the bad channels file
    patient_id : str
        Patient identifier for logging
    movie_base : str
        Movie base name for logging
        
    Returns:
    --------
    bool
        True if successful, False otherwise
    """
    try:
        bad_channels = raw_data.info['bads']
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        # Save bad channels to file
        with open(file_path, 'w') as f:
            f.write(f"# Bad channels for {patient_id} - {movie_base}\n")
            f.write(f"# Total channels: {len(raw_data.ch_names)}\n")
            f.write(f"# Bad channels: {len(bad_channels)}\n")
            f.write("# Channel names:\n")
            for ch in bad_channels:
                f.write(f"{ch}\n")
        
        print(f"✓ Bad channels saved to: {file_path}")
        print(f"  - {len(bad_channels)} bad channels out of {len(raw_data.ch_names)} total")
        return True
        
    except Exception as e:
        print(f"✗ Error saving bad channels: {e}")
        return False

def load_bad_channels(file_path, raw_data, patient_id, movie_base):
    """
    Load bad channels from a text file and apply them to the MNE Raw object.
    
    Parameters:
    -----------
    file_path : str
        Path to the bad channels file
    raw_data : mne.io.Raw
        The MNE Raw object to apply bad channels to
    patient_id : str
        Patient identifier for logging
    movie_base : str
        Movie base name for logging
        
    Returns:
    --------
    mne.io.Raw
        The MNE Raw object with bad channels applied
    """
    try:
        if not os.path.exists(file_path):
            print(f"⚠️ No bad channels file found at: {file_path}")
            return raw_data
        
        # Read bad channels from file
        bad_channels = []
        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    bad_channels.append(line)
        
        # Verify that all bad channels exist in the current data
        valid_bad_channels = [ch for ch in bad_channels if ch in raw_data.ch_names]
        invalid_bad_channels = [ch for ch in bad_channels if ch not in raw_data.ch_names]
        
        if invalid_bad_channels:
            print(f"⚠️ Some bad channels from file not found in current data: {invalid_bad_channels}")
        
        # Apply bad channels to the data
        raw_data.info['bads'] = valid_bad_channels
        
        print(f"✓ Bad channels loaded from: {file_path}")
        print(f"  - {len(valid_bad_channels)} valid bad channels applied")
        if invalid_bad_channels:
            print(f"  - {len(invalid_bad_channels)} invalid channels ignored")
        
        return raw_data
        
    except Exception as e:
        print(f"✗ Error loading bad channels: {e}")
        return raw_data

def check_bad_channels_integrity(raw_data, stage_name):
    """
    Check and report the integrity of bad channel information.
    
    Parameters:
    -----------
    raw_data : mne.io.Raw
        The MNE Raw object to check
    stage_name : str
        Name of the processing stage for logging
        
    Returns:
    --------
    dict
        Dictionary with integrity check results
    """
    total_channels = len(raw_data.ch_names)
    bad_channels = raw_data.info['bads']
    good_channels = [ch for ch in raw_data.ch_names if ch not in bad_channels]
    
    # Check for consistency
    bad_channel_consistency = all(ch in raw_data.ch_names for ch in bad_channels)
    
    result = {
        'total_channels': total_channels,
        'bad_channels': len(bad_channels),
        'good_channels': len(good_channels),
        'bad_channel_names': bad_channels,
        'good_channel_names': good_channels,
        'consistency_check': bad_channel_consistency,
        'bad_channel_percentage': (len(bad_channels) / total_channels) * 100 if total_channels > 0 else 0
    }
    
    print(f"\n--- Bad Channel Integrity Check ({stage_name}) ---")
    print(f"Total channels: {result['total_channels']}")
    print(f"Bad channels: {result['bad_channels']} ({result['bad_channel_percentage']:.1f}%)")
    print(f"Good channels: {result['good_channels']}")
    
    if not bad_channel_consistency:
        print("⚠️ WARNING: Some bad channels are not in the channel list!")
    
    return result

def validate_mne_structure(raw_data, stage_name):
    """
    Validate that MNE data structure is intact and complete.
    
    Parameters:
    -----------
    raw_data : mne.io.Raw
        The MNE Raw object to validate
    stage_name : str
        Name of the processing stage for logging
        
    Returns:
    --------
    dict
        Dictionary with validation results
    """
    validation_results = {
        'has_montage': raw_data.info['dig'] is not None,
        'has_bad_channels': hasattr(raw_data.info, 'bads'),
        'has_channel_info': hasattr(raw_data.info, 'chs'),
        'has_sampling_rate': hasattr(raw_data.info, 'sfreq'),
        'has_data': hasattr(raw_data, '_data'),
        'channel_count': len(raw_data.ch_names) if hasattr(raw_data, 'ch_names') else 0,
        'data_shape': raw_data._data.shape if hasattr(raw_data, '_data') else None
    }
    
    print(f"\n--- MNE Structure Validation ({stage_name}) ---")
    print(f"✓ Has montage: {validation_results['has_montage']}")
    print(f"✓ Has bad channels: {validation_results['has_bad_channels']}")
    print(f"✓ Has channel info: {validation_results['has_channel_info']}")
    print(f"✓ Has sampling rate: {validation_results['has_sampling_rate']}")
    print(f"✓ Has data: {validation_results['has_data']}")
    print(f"✓ Channel count: {validation_results['channel_count']}")
    print(f"✓ Data shape: {validation_results['data_shape']}")
    
    # Check for critical issues
    issues = []
    if not validation_results['has_data']:
        issues.append("No data array found")
    if not validation_results['has_channel_info']:
        issues.append("No channel information found")
    if not validation_results['has_sampling_rate']:
        issues.append("No sampling rate found")
    
    if issues:
        print(f"⚠️ WARNING: Structure issues found: {issues}")
        validation_results['has_issues'] = True
        validation_results['issues'] = issues
    else:
        print("✓ MNE structure is intact")
        validation_results['has_issues'] = False
        validation_results['issues'] = []
    
    return validation_results

def preserve_mne_structure(raw_data, operation_name):
    """
    Ensure MNE data structure is preserved during operations.
    
    Parameters:
    -----------
    raw_data : mne.io.Raw
        The MNE Raw object
    operation_name : str
        Name of the operation being performed
        
    Returns:
    --------
    mne.io.Raw
        The MNE Raw object with structure preserved
    """
    # Validate structure before operation
    pre_validation = validate_mne_structure(raw_data, f"Before {operation_name}")
    
    if pre_validation['has_issues']:
        print(f"⚠️ Structure issues detected before {operation_name}")
        return raw_data
    
    # Store critical information
    original_montage = raw_data.info['dig']
    original_bad_channels = raw_data.info['bads'].copy() if raw_data.info['bads'] else []
    original_sfreq = raw_data.info['sfreq']
    
    # Return the data (operations should be applied externally)
    # This function serves as a checkpoint for structure validation
    return raw_data

def restore_mne_structure(raw_data, original_montage, original_bad_channels, original_sfreq, operation_name):
    """
    Restore MNE data structure after operations.
    
    Parameters:
    -----------
    raw_data : mne.io.Raw
        The MNE Raw object after operation
    original_montage : list
        Original montage information
    original_bad_channels : list
        Original bad channels list
    original_sfreq : float
        Original sampling frequency
    operation_name : str
        Name of the operation that was performed
        
    Returns:
    --------
    mne.io.Raw
        The MNE Raw object with structure restored
    """
    # Restore montage if it was lost
    if raw_data.info['dig'] is None and original_montage is not None:
        raw_data.info['dig'] = original_montage
        print(f"✓ Restored montage after {operation_name}")
    
    # Restore bad channels if they were lost
    if not raw_data.info['bads'] and original_bad_channels:
        raw_data.info['bads'] = original_bad_channels
        print(f"✓ Restored {len(original_bad_channels)} bad channels after {operation_name}")
    
    # Validate structure after restoration
    post_validation = validate_mne_structure(raw_data, f"After {operation_name}")
    
    return raw_data

def apply_highpass_filter(raw_data, sfreq, l_freq=0.5, method="fir", verbose=True):
    """
    Apply high-pass filter to remove drift from EEG data.
    
    Parameters:
    -----------
    raw_data : mne.io.Raw
        The MNE Raw object containing the EEG data
    sfreq : float
        Sampling frequency of the data
    l_freq : float, optional
        High-pass frequency cutoff (default: 0.5 Hz)
    method : str, optional
        Filter method (default: "fir")
    verbose : bool, optional
        Whether to print filter details (default: True)
        
    Returns:
    --------
    mne.io.Raw
        The high-pass filtered Raw object
    """
    print(f'--->Applying high-pass filter at {l_freq} Hz to remove drift')
    
    # Preserve structure before filtering
    raw_data = preserve_mne_structure(raw_data, "Highpass Filter")
    original_montage = raw_data.info['dig']
    original_bad_channels = raw_data.info['bads'].copy() if raw_data.info['bads'] else []
    original_sfreq = raw_data.info['sfreq']
    
    # Apply the high-pass filter
    filtered_data = filter_data(
        data=raw_data.get_data(),
        sfreq=sfreq,
        l_freq=l_freq,
        h_freq=None,  # No low-pass filter
        method=method,
        fir_window='hamming',
        fir_design='firwin',
        verbose=verbose
    )
    
    # Create new MNE Raw object with filtered data
    filtered_raw = raw_data.copy()
    filtered_raw._data = filtered_data
    
    # Restore structure after filtering
    filtered_raw = restore_mne_structure(
        filtered_raw, original_montage, original_bad_channels, original_sfreq, "Highpass Filter"
    )
    
    return filtered_raw

def create_file_paths(patient_id, implant_id, movie_filename, prep_dir, neural_prep_dir, hfa_dir):
    """
    Create standardized file paths for the preprocessing pipeline.
    
    Parameters:
    -----------
    patient_id : str
        Patient identifier (e.g., 'NS190')
    implant_id : str
        Implant identifier (e.g., 'implant01')
    movie_filename : str
        Full movie filename with .nwb extension (e.g., 'despicable_me_english.nwb')
    prep_dir : str
        Base preprocessing directory
    neural_prep_dir : str
        Neural preprocessing subdirectory name
    hfa_dir : str
        HFA processing subdirectory name
        
    Returns:
    --------
    dict
        Dictionary containing all file paths with standardized naming:
        - movie_base: movie name without extension
        - sub_prep_dir: subject-specific preprocessing directory
        - sub_hfa_dir: subject-specific HFA directory
        - bad_channels_file: path to bad channels file
        - wm_contacts_file: path to white matter contacts file
        - preprocessed_file: path to preprocessed file (after filtering/downsampling)
        - referenced_files: dict of referenced files by reference type
        - hfa_files: dict of HFA files by reference type
    """
    
    # Extract movie base name (without .nwb extension)
    movie_base = movie_filename.replace('.nwb', '')
    
    # Create subject-specific directories
    sub_prep_dir = os.path.join(prep_dir, patient_id, neural_prep_dir)
    sub_hfa_dir = os.path.join(prep_dir, patient_id, hfa_dir)
    
    # Create standardized file paths
    file_paths = {
        'movie_base': movie_base,
        'sub_prep_dir': sub_prep_dir,
        'sub_hfa_dir': sub_hfa_dir,
        'bad_channels_file': os.path.join(sub_prep_dir, f"{movie_base}_bad_channels.txt"),
        'wm_contacts_file': os.path.join(sub_prep_dir, f"{movie_base}_wm_contacts.txt"),
        'preprocessed_file': os.path.join(sub_prep_dir, f"{movie_base}_preprocessed.fif"),
        'referenced_files': {},
        'hfa_files': {}
    }
    
    # Create reference-specific file paths
    ref_types = ['avg', 'wm', 'wm_avg', 'bip','wm_bip','wm_legacy']
    for ref in ref_types:
        file_paths['referenced_files'][ref] = os.path.join(
            sub_prep_dir, f"{movie_base}_referenced_{ref}.fif"
        )
        file_paths['hfa_files'][ref] = os.path.join(
            sub_hfa_dir, f"{movie_base}_hfa_{ref}.fif"
        )
    
    return file_paths

def plot_power_spectra(data, labels, fs_data, freq_range, pat, plot_title="Power Spectral Density (Welch)", 
                       x_scale='linear', y_scale='linear'):
    """
    Compute and plot Power Spectral Density (PSD) for given data using Welch's method.

    Parameters:
    - data (ndarray): The bandpass-filtered data array with shape (n_channels, n_times).
    - labels (list): List of channel labels.
    - fs_data (float): Sampling frequency of the data.
    - freq_range (tuple): Frequency range for the PSD (fmin, fmax).
    - pat (str): Identifier for the patient or dataset.
    - plot_title (str): Title for the plot.
    - x_scale (str): Scale for x-axis ('linear' or 'log').
    - y_scale (str): Scale for y-axis ('linear' or 'log').

    Returns:
    - fig: The matplotlib figure object of the PSD plot.
    """
    
    # Compute the PSD using Welch's method
    psd, freqs = psd_array_welch(
        data, sfreq=fs_data, fmin=freq_range[0], fmax=freq_range[1],
        n_fft=int(fs_data * 2), n_overlap=int(fs_data), average='mean'
    )
    
    # Plot the PSD for each channel
    fig, ax = plt.subplots(figsize=(10, 6))
    for i, label in enumerate(labels):
        ax.plot(freqs, psd[i, :], label=label)
    
    # Set scales based on parameters
    if x_scale == 'log':
        ax.set_xscale('log')
    if y_scale == 'log':
        ax.set_yscale('log')
    
    # Add labels and legend
    ax.set_title(f'{pat} {freq_range} {plot_title}', fontsize=14)
    ax.set_xlabel('Frequency (Hz)', fontsize=12)
    ax.set_ylabel('Power Spectral Density (dB)', fontsize=12)
    ax.grid(True)
    ax.legend(loc='upper right', fontsize='small')
    plt.tight_layout()
    plt.show()
    
    return fig


def load_preprocessed_data(file_path, bad_channels_file, patient_id, movie_base):
    """
    Load preprocessed data and restore bad channel information.
    
    Parameters:
    -----------
    file_path : str
        Path to the preprocessed .fif file
    bad_channels_file : str
        Path to the bad channels file
    patient_id : str
        Patient identifier for logging
    movie_base : str
        Movie base name for logging
        
    Returns:
    --------
    mne.io.Raw
        The loaded MNE Raw object with bad channels restored
    """
    try:
        # Load the preprocessed data
        raw_data = mne.io.read_raw_fif(file_path, preload=True)
        print(f"✓ Preprocessed data loaded from: {file_path}")
        
        # Load and apply bad channels
        raw_data = load_bad_channels(bad_channels_file, raw_data, patient_id, movie_base)
        
        # Check integrity
        check_bad_channels_integrity(raw_data, "Loaded Preprocessed Data")
        
        return raw_data
        
    except Exception as e:
        print(f"✗ Error loading preprocessed data: {e}")
        return None

def summarize_bad_channels(raw_data, stage_name):
    """
    Print a summary of bad channel status.
    
    Parameters:
    -----------
    raw_data : mne.io.Raw
        The MNE Raw object to check
    stage_name : str
        Name of the processing stage for logging
    """
    total_channels = len(raw_data.ch_names)
    bad_channels = raw_data.info['bads']
    good_channels = [ch for ch in raw_data.ch_names if ch not in bad_channels]
    
    print(f"\n--- Bad Channel Summary ({stage_name}) ---")
    print(f"Total channels: {total_channels}")
    print(f"Bad channels: {len(bad_channels)} ({len(bad_channels)/total_channels*100:.1f}%)")
    print(f"Good channels: {len(good_channels)}")
    if bad_channels:
        print(f"Bad channel names: {bad_channels}")
    else:
        print("No bad channels marked")
    print("-" * 50)

def load_data_with_bad_channels(file_path, bad_channels_file, patient_id, movie_base):
    """
    Load data and restore bad channel information.
    
    Parameters:
    -----------
    file_path : str
        Path to the data file (.fif)
    bad_channels_file : str
        Path to the bad channels file
    patient_id : str
        Patient identifier for logging
    movie_base : str
        Movie base name for logging
        
    Returns:
    --------
    mne.io.Raw
        The loaded MNE Raw object with bad channels restored
    """
    try:
        # Load the data
        raw_data = mne.io.read_raw_fif(file_path, preload=True)
        print(f"✓ Data loaded from: {file_path}")
        
        # Print initial bad channel status
        print(f"Initial bad channels in loaded data: {raw_data.info['bads']}")
        
        # Load and apply bad channels
        raw_data = load_bad_channels(bad_channels_file, raw_data, patient_id, movie_base)
        
        # Print final bad channel status
        print(f"Final bad channels after restoration: {raw_data.info['bads']}")
        
        # Summarize bad channel status
        summarize_bad_channels(raw_data, "After Loading and Restoration")
        
        # Check integrity
        check_bad_channels_integrity(raw_data, "Loaded Data")
        
        return raw_data
        
    except Exception as e:
        print(f"✗ Error loading data: {e}")
        return None

class ProcessingLogger:
    """
    Minimal logging class for tracking processing decisions.
    """
    
    def __init__(self, patient_id, movie_name, log_dir):
        """
        Initialize the logger.
        
        Parameters:
        -----------
        patient_id : str
            Patient identifier
        movie_name : str
            Movie name
        log_dir : str
            Directory to save log files
        """
        from datetime import datetime
        
        self.patient_id = patient_id
        self.movie_name = movie_name
        self.log_dir = log_dir
        self.start_time = datetime.now()
        self.log_entries = []
        
        # Create log file path with timestamp
        os.makedirs(log_dir, exist_ok=True)
        timestamp = self.start_time.strftime('%Y%m%d_%H%M%S')
        self.log_file = os.path.join(log_dir, f"{patient_id}_{movie_name}_processing_log_{timestamp}.txt")
        
        # Initialize log file
        self._write_header()
    
    def _write_header(self):
        """Write the initial header to the log file."""
        from datetime import datetime
        
        header = f"""================================================================
        PROCESSING LOG: {self.patient_id} - {self.movie_name}
        ================================================================
        Timestamp: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}
        Patient: {self.patient_id}
        Movie: {self.movie_name}
        Log file: {self.log_file}
        ================================================================
        
        """
        
        with open(self.log_file, 'w') as f:
            f.write(header)
    
    def log_section(self, section_name):
        """Log a new section header."""
        section = f"""
        ================================================================
        {section_name.upper()}
        ================================================================
        """
        with open(self.log_file, 'a') as f:
            f.write(section)
    
    def log_decision(self, decision, details):
        """Log a single decision with details."""
        from datetime import datetime
        timestamp = datetime.now().strftime('%H:%M:%S')
        
        entry = f"[{timestamp}] {decision}: {details}\n"
        
        with open(self.log_file, 'a') as f:
            f.write(entry)
    
    def log_filtering(self, filter_type, parameters, status="Applied"):
        """Log filtering decisions."""
        self.log_decision(f"{filter_type.upper()} FILTER", f"{parameters} - Status: {status}")
    
    def log_bad_channels(self, bad_channels, method="Manual"):
        """Log bad channel selection."""
        self.log_decision("BAD CHANNELS", f"Method: {method}, Count: {len(bad_channels)}, Channels: {bad_channels}")
    
    def log_referencing(self, ref_type, details):
        """Log referencing decisions."""
        self.log_decision(f"{ref_type.upper()} REFERENCING", details)
    
    def log_wm_selection(self, wm_contacts, selection_criteria=""):
        """Log white matter contact selection."""
        self.log_decision("WM CONTACTS", f"Selected: {wm_contacts}, Criteria: {selection_criteria}")
    
    def log_file_save(self, file_type, file_path):
        """Log file saving operations."""
        self.log_decision("FILE SAVE", f"{file_type}: {file_path}")
    
    def log_nwb_load(self, nwb_file_path):
        """Log NWB file loading operations."""
        self.log_decision("NWB LOAD", f"Loading NWB file: {nwb_file_path}")
    
    def log_fif_save(self, fif_file_path, file_description="FIF file"):
        """Log FIF file saving operations."""
        self.log_decision(f"{file_description} SAVE", f"{file_description}: {fif_file_path}")
    
    def log_error(self, error_msg):
        """Log errors or warnings."""
        self.log_decision("ERROR/WARNING", error_msg)
    
    def log_quality_metrics(self, metrics_dict):
        """Log quality metrics."""
        self.log_section("QUALITY METRICS")
        for metric, value in metrics_dict.items():
            self.log_decision(metric, str(value))
    
    def finish_log(self):
        """Write the end of processing summary."""
        from datetime import datetime
        
        end_time = datetime.now()
        duration = end_time - self.start_time
        
        footer = f"""
================================================================
END OF PROCESSING
================================================================
Start Time: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}
End Time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}
Duration: {duration}
Status: COMPLETED
================================================================
"""
        
        with open(self.log_file, 'a') as f:
            f.write(footer)

def plot_psd_batched(raw_data, batch_size=32, freq_bands=None, patient_id="Patient", 
                     x_scale='linear', y_scale='linear'):
    """
    Create PSD plots using the original approach but in batches of channels.
    Uses the same filtering and plotting style as the original plot_power_spectra.
    
    Parameters:
    -----------
    raw_data : mne.io.Raw
        The MNE Raw object containing the data
    batch_size : int, optional
        Number of channels per batch (default: 32)
    freq_bands : list, optional
        List of frequency band names. Default: ['low', 'middle', 'high']
    patient_id : str, optional
        Patient identifier for plot titles
    x_scale : str, optional
        Scale for x-axis ('linear' or 'log', default: 'linear')
    y_scale : str, optional
        Scale for y-axis ('linear' or 'log', default: 'linear')
        
    Returns:
    --------
    list
        List of figure objects created
    """
    if freq_bands is None:
        freq_bands = ['low', 'middle', 'high']
    
    # Get data and parameters
    data = raw_data.get_data()  # Shape: (n_channels, n_samples)
    fs_data = raw_data.info['sfreq']
    
    # Get good channel indices
    labels = list(raw_data.ch_names)
    good_ch_indices = [i for i, ch in enumerate(labels) if ch not in raw_data.info['bads']]
    
    # Filter the data to include only good channels
    data_good = data[good_ch_indices, :]
    labels_good = [labels[i] for i in good_ch_indices]
    
    total_good_channels = len(labels_good)
    if total_good_channels == 0:
        print("⚠️ No good channels available")
        return []
    
    # Calculate number of batches
    num_batches = (total_good_channels + batch_size - 1) // batch_size  # Ceiling division
    
    print(f"\n--- Batched PSD Visualization ---")
    print(f"Total good channels: {total_good_channels}")
    print(f"Batch size: {batch_size}")
    print(f"Number of batches: {num_batches}")
    print(f"Frequency bands: {freq_bands}")
    print("-" * 50)
    
    figures = []
    
    # Process each frequency band
    for freq_band in freq_bands:
        if freq_band == 'low':
            freq_range = (0.5, 7)
        elif freq_band == 'middle':
            freq_range = (8, 30)
        elif freq_band == 'high':
            freq_range = (30, 170)
        elif freq_band == 'all':
            freq_range = (.5,200)
        else:
            print(f"⚠️ Unknown frequency band: {freq_band}")
            continue
        
        print(f"\n--- Processing {freq_band} band ({freq_range[0]}-{freq_range[1]} Hz) ---")
        
        # Bandpass filter the data for this frequency band
        sos = signal.butter(5, list(freq_range), btype='bandpass', output='sos', fs=fs_data)
        band_dat = signal.sosfiltfilt(sos, data_good, axis=1)
        
        # Create batches for this frequency band
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, total_good_channels)
            
            # Get subset of data and labels for this batch
            batch_data = band_dat[start_idx:end_idx, :]
            batch_labels = labels_good[start_idx:end_idx]
            
            print(f"  Batch {batch_idx + 1}/{num_batches}: Channels {start_idx + 1}-{end_idx}")
            
            # Create plot using the original plot_power_spectra approach
            fig = plot_power_spectra(
                batch_data, 
                batch_labels, 
                fs_data, 
                freq_range, 
                patient_id, 
                plot_title=f"Power Spectral Density (Welch) for {freq_band} - Batch {batch_idx + 1}/{num_batches} (Channels {start_idx + 1}-{end_idx})",
                x_scale=x_scale,
                y_scale=y_scale
            )
            figures.append(fig)
    
    print(f"\n✅ Completed {len(figures)} total plots")
    print(f"  - {len(freq_bands)} frequency bands")
    print(f"  - {num_batches} batches per band")
    
    return figures


def plot_psd_with_scales(raw_data, scale_type='linear', batch_size=32, freq_bands=None, patient_id="Patient"):
    """
    Convenience function to create PSD plots with common scale combinations.
    
    Parameters:
    -----------
    raw_data : mne.io.Raw
        The MNE Raw object containing the data
    scale_type : str, optional
        Scale type ('linear', 'log_x', 'log_y', 'log_both', default: 'linear')
    batch_size : int, optional
        Number of channels per batch (default: 32)
    freq_bands : list, optional
        List of frequency band names. Default: ['low', 'middle', 'high']
    patient_id : str, optional
        Patient identifier for plot titles
        
    Returns:
    --------
    list
        List of figure objects created
    """
    # Map scale_type to x_scale and y_scale
    scale_mapping = {
        'linear': ('linear', 'linear'),
        'log_x': ('log', 'linear'),
        'log_y': ('linear', 'log'),
        'log_both': ('log', 'log')
    }
    
    if scale_type not in scale_mapping:
        print(f"⚠️ Unknown scale_type: {scale_type}. Using 'linear' instead.")
        scale_type = 'linear'
    
    x_scale, y_scale = scale_mapping[scale_type]
    
    print(f"Creating PSD plots with scale: {scale_type} (x: {x_scale}, y: {y_scale})")
    
    return plot_psd_batched(
        raw_data, 
        batch_size=batch_size, 
        freq_bands=freq_bands, 
        patient_id=patient_id,
        x_scale=x_scale,
        y_scale=y_scale
    )

import re
from collections import defaultdict


def regress_out_noise_by_group(raw, group_def):
    """
    Regress group-specific noise channels out of specified channels in an MNE Raw.

    Parameters
    ----------
    raw : mne.io.BaseRaw
        Raw object (will be copied).
    group_def : dict
        Dictionary of the form:
        {
          "group_name": {
              "noise": "NoiseChannelName",
              "channels": ["ch1", "ch2", ...]
          },
          ...
        }

    Returns
    -------
    raw_clean : mne.io.BaseRaw
        New Raw object with noise regressed out per group.
    """
    raw_clean = raw.copy()
    data = raw_clean.get_data()  # shape: (n_channels, n_times)
    ch_names = raw_clean.ch_names

    for group_name, info in group_def.items():
        noise_ch = info["noise"]
        group_chs = info["channels"]

        if noise_ch not in ch_names:
            raise ValueError(f"Noise channel {noise_ch} for {group_name} not found in raw.ch_names")

        noise_idx = ch_names.index(noise_ch)
        noise = data[noise_idx, :].copy()
        noise = noise - noise.mean()  # demean

        denom = np.dot(noise, noise)
        if denom == 0:
            raise ValueError(
                f"Noise channel {noise_ch} for {group_name} appears to be flat (all zeros)."
            )

        # Restrict to channels that exist in raw
        group_indices = [ch_names.index(ch) for ch in group_chs if ch in ch_names]

        for ch_idx in group_indices:
            if ch_idx == noise_idx:
                # leave the noise channel itself untouched (you'll drop it later)
                continue

            y = data[ch_idx, :]
            y_demean = y - y.mean()

            beta = np.dot(y_demean, noise) / denom
            y_clean = y_demean - beta * noise

            # restore original mean (optional)
            data[ch_idx, :] = y_clean + y.mean()

    raw_clean._data = data
    return raw_clean


def make_groups_from_prefix(raw, min_group_size=2):
    groups = defaultdict(list)
    for ch in raw.ch_names:
        if ch in raw.info['bads']:
            continue
        m = re.match(r'([A-Za-z]+)\d+', ch)
        if m:
            prefix = m.group(1)  # e.g., 'LTG' from 'LTG1'
            groups[prefix].append(ch)

    # drop tiny groups if desired
    groups = {g: chs for g, chs in groups.items() if len(chs) >= min_group_size}
    return groups


def reref_avg_by_group(raw, groups, use_only_good=True):
    """
    Apply group-wise average reference.

    Parameters
    ----------
    raw : mne.io.Raw
        Raw object to re-reference (will be copied).
    groups : dict
        Mapping {group_name: [ch1, ch2, ...]} defining each depth/bank/grid.
    use_only_good : bool
        If True, channels in raw.info['bads'] are excluded from the mean
        but can still have the reference subtracted (optional behavior).

    Returns
    -------
    raw_ref : mne.io.Raw
        New Raw object with group-wise average reference applied.
    """
    raw_ref = raw.copy()
    ch_names = raw_ref.ch_names
    bads = set(raw_ref.info['bads']) if use_only_good else set()

    for gname, group_chs in groups.items():
        # channels that actually exist in this Raw
        group_chs = [ch for ch in group_chs if ch in ch_names]
        if len(group_chs) == 0:
            continue

        # channels used to compute the mean (exclude bads if requested)
        mean_chs = [ch for ch in group_chs if ch not in bads]
        if len(mean_chs) < 2:
            # skip if not enough good channels to form a stable mean
            continue

        idx_all  = mne.pick_channels(ch_names, include=group_chs)
        idx_mean = mne.pick_channels(ch_names, include=mean_chs)

        # compute group average over time
        group_mean = raw_ref._data[idx_mean, :].mean(axis=0, keepdims=True)

        # subtract mean from all channels in the group
        raw_ref._data[idx_all, :] -= group_mean

    return raw_ref



def detect_spikes_ref1(raw, ref_name='Ref1',
                       thresh_z=8.0,
                       max_width_ms=20.0,
                       min_separation_ms=5.0):
    """
    Detect brief spikes on the Ref1 channel based on the temporal derivative.

    Parameters
    ----------
    raw : mne.io.Raw
        Raw object (already filtered as you want).
    ref_name : str
        Name of the reference channel (e.g., 'Ref1').
    thresh_z : float
        Z-score threshold on the derivative to flag candidate spikes.
    max_width_ms : float
        Maximum duration (in ms) of a spike event.
    min_separation_ms : float
        Minimum separation between consecutive spikes (to avoid double-counting).

    Returns
    -------
    spike_samples : np.ndarray
        Array of sample indices (int) representing spike centers.
    """
    sfreq = raw.info['sfreq']
    max_width_samp = int(max_width_ms * sfreq / 1000.0)
    min_sep_samp = int(min_separation_ms * sfreq / 1000.0)

    # Get Ref1 data (first row)
    ref_data = raw.get_data(picks=[ref_name])[0]

    # First derivative
    diff = np.diff(ref_data)

    # Robust z-scoring of derivative
    med = np.median(diff)
    mad = np.median(np.abs(diff - med))
    # handle potential zero MAD
    if mad == 0:
        raise RuntimeError("MAD is zero for Ref1 derivative; check your data / preprocessing.")

    z = 0.6745 * (diff - med) / mad

    # Candidate indices where derivative is extreme
    cand_idx = np.where(np.abs(z) > thresh_z)[0]

    if cand_idx.size == 0:
        return np.array([], dtype=int)

    # Group contiguous indices into events
    events = []
    current = [cand_idx[0]]
    for idx in cand_idx[1:]:
        if idx == current[-1] + 1:
            current.append(idx)
        else:
            events.append(current)
            current = [idx]
    events.append(current)

    # Keep only short events (true spikes)
    spike_centers = []
    last_center = -np.inf

    for ev in events:
        width = ev[-1] - ev[0] + 1
        if width <= max_width_samp:
            center = (ev[0] + ev[-1]) // 2
            # Enforce minimum separation between spike centers
            if center - last_center >= min_sep_samp:
                spike_centers.append(center)
                last_center = center

    # Note: diff is length N-1, so shift indices by 1 to map to raw samples
    spike_samples = np.array(spike_centers, dtype=int) + 1
    return spike_samples


import numpy as np

def detect_spikes_all_channels(raw,
                               picks=None,
                               thresh_z=8.0,
                               max_width_ms=20.0,
                               min_separation_ms=5.0):
    """
    Detect brief spikes on each channel based on the temporal derivative,
    using the same logic as `detect_spikes_ref1`.

    Parameters
    ----------
    raw : mne.io.Raw
        Raw object (already filtered as you want).
    picks : list | None
        Channels to analyze. Can be a list of channel names or indices.
        If None, all data channels in `raw` are used.
    thresh_z : float
        Z-score threshold on the derivative to flag candidate spikes.
    max_width_ms : float
        Maximum duration (in ms) of a spike event.
    min_separation_ms : float
        Minimum separation between consecutive spikes (to avoid double-counting).

    Returns
    -------
    spikes_by_chan : dict
        Dictionary mapping channel name -> np.ndarray of sample indices
        representing spike centers for that channel.
    """
    sfreq = raw.info['sfreq']
    max_width_samp = int(max_width_ms * sfreq / 1000.0)
    min_sep_samp   = int(min_separation_ms * sfreq / 1000.0)

    # Resolve picks
    if picks is None:
        picks = mne.pick_types(raw.info, meg=False, eeg=True, seeg=True, ecog=True, misc=False)
    else:
        # Let mne handle flexible picks (names or indices)
        picks = mne.pick_channels(raw.info['ch_names'], include=picks)

    spikes_by_chan = {}

    for pick in picks:
        ch_name = raw.info['ch_names'][pick]
        data = raw.get_data(picks=[pick])[0]  # 1D array

        # First derivative
        diff = np.diff(data)

        # Robust z-scoring of derivative
        med = np.median(diff)
        mad = np.median(np.abs(diff - med))

        if mad == 0:
            # If the channel is flat / no variability, just skip
            spikes_by_chan[ch_name] = np.array([], dtype=int)
            continue

        z = 0.6745 * (diff - med) / mad

        # Candidate indices where derivative is extreme
        cand_idx = np.where(np.abs(z) > thresh_z)[0]

        if cand_idx.size == 0:
            spikes_by_chan[ch_name] = np.array([], dtype=int)
            continue

        # Group contiguous indices into events
        events = []
        current = [cand_idx[0]]
        for idx in cand_idx[1:]:
            if idx == current[-1] + 1:
                current.append(idx)
            else:
                events.append(current)
                current = [idx]
        events.append(current)

        # Keep only short events (true spikes)
        spike_centers = []
        last_center = -np.inf

        for ev in events:
            width = ev[-1] - ev[0] + 1
            if width <= max_width_samp:
                center = (ev[0] + ev[-1]) // 2
                # Enforce minimum separation between spike centers
                if center - last_center >= min_sep_samp:
                    spike_centers.append(center)
                    last_center = center

        # Note: diff is length N-1, so shift indices by 1 to map to raw samples
        spike_samples = np.array(spike_centers, dtype=int) + 1
        spikes_by_chan[ch_name] = spike_samples

    return spikes_by_chan


def interpolate_spikes(raw, spike_samples,
                       window_ms=10.0):
    """
    Interpolate over spike windows across all channels.

    Parameters
    ----------
    raw : mne.io.Raw
        Raw object; will be modified in-place unless you copy() before.
    spike_samples : array-like
        Sample indices of spike centers (e.g., from detect_spikes_ref1).
    window_ms : float
        Half-width of interpolation window in ms (i.e., +/- window_ms around center).

    Returns
    -------
    raw_clean : mne.io.Raw
        Raw object with spikes interpolated (same object as input unless copied).
    """
    raw_clean = raw  # modify in-place, or use raw.copy() if you prefer
    sfreq = raw_clean.info['sfreq']
    half_win_samp = int(window_ms * sfreq / 1000.0)

    data = raw_clean._data  # shape (n_channels, n_times)
    n_times = data.shape[1]

    for center in spike_samples:
        start = max(center - half_win_samp, 1)       # avoid index 0 edge
        end   = min(center + half_win_samp, n_times - 2)  # avoid last index

        # values just before and after the window
        left_idx = start - 1
        right_idx = end + 1

        if left_idx < 0 or right_idx >= n_times:
            continue  # skip spikes too close to edges

        # Linear interpolation between left_idx and right_idx
        for ch in range(data.shape[0]):
            y0 = data[ch, left_idx]
            y1 = data[ch, right_idx]
            n  = end - start + 1
            interp_vals = np.linspace(y0, y1, n, endpoint=True)
            data[ch, start:end+1] = interp_vals

    return raw_clean


import os
import numpy as np
import mne
from mne import EvokedArray, create_info
from tqdm import tqdm



def filter_hfa_epochs(epochs,filter_bank=None,exclude=(60,120,180),exclude_width=1,method='hilbert',n_jobs=1,baseline_time=None, resample_fs=None):
    """Extract HFA from epochs

    Args:
        epochs: mne.Epochs object
        filter_bank: List or tuple containing 2 numbers for bandpass filters
        exclude: Frequency bands to exclude (NOT READY)
        exclude_width: Width of frequency bands to exclude (NOT READY)
        method: Only 'hilbert' currently available
        n_jobs: Number of threads that can be used at once
        baseline_time: tuple, list or numpy.array indicates time for baseline correction
        resample_fs: frequency to resample to after filtering (makes the HFA analysis more managable)

    Returns: list | mne.Evoked object

    Examples:
        hfas = filter_hfa(epochs, n_jobs=2, baseline_time=(-0.35, -0.05))

    """

    # Filter bank settings
    if filter_bank == None:
        min_freq_filter_bank = np.arange(70,170,10)
        max_freq_filter_bank = np.arange(80,170+1,10)
        filter_bank = list(zip(min_freq_filter_bank,max_freq_filter_bank))

    ###############################################################
    # This section is for excluding particular frequencies
    # Not yet ready
    # Make the exclusion frequencies if requested
    # exclude_freq_sets = []
    # if type(exclude) in [list, tuple, np.array]:
    #     exclude_freqs = np.array(exclude)
    #     if type(exclude_width) in [float,int]:
    #         min_ex_freqs = exclude_freqs - exclude_width
    #         max_ex_freqs = exclude_freqs + exclude_width
    #         exclude_freqs = np.unique( np.concatenate((min_ex_freqs, exclude_freqs, max_ex_freqs)) )
    # else:
    #     exclude_freqs = np.array([])
    #
    #
    # # Create the full set of numbers each set encompasses
    # frequency_sets = []
    # for freqs in filter_bank:
    #     this_f_range = np.arange(freqs[0], freqs[1] + 1, 1)
    #     this_f_range = np.setdiff1d(this_f_range, exclude_freqs)
    #     frequency_sets.append(this_f_range)
    ###############################################################

    # Loop over each separate event type epoch
    event_types = list(epochs.event_id.keys())
    hfa_list = []

    if resample_fs != None:
        info_fs = resample_fs
    else:
        info_fs = epochs.info['sfreq']

    # Info object to use after
    ch_types = epochs.info.get_channel_types()
    hfa_info = create_info(epochs.ch_names, info_fs, ch_types)
    montage = epochs.get_montage()

    # Initiate a progress bar to track what's going on and how long it will take
    pbar = tqdm(total=len(event_types)*len(filter_bank), ncols=100, desc='Running filter_hfa')

    for evt in event_types:
        evt_epoch = epochs[evt]
        evt_epoch.load_data()

        # Loop through frequency bands by steps
        freq_band_means = []
        for fband in filter_bank:

            # Output is: TRIALxCHANNELxTIME
            gamma_fband = evt_epoch.copy().filter(fband[0], fband[1], verbose='ERROR').apply_hilbert(envelope=True, n_jobs=n_jobs)

            if resample_fs != None:
                gamma_fband.resample(resample_fs)

            # Get data and normalize it by dividing by mean over time
            gamma_fband_data = gamma_fband.get_data()
            gamma_fband_data_time_mean = gamma_fband_data.mean(axis=2)
            gamma_fband_data_time_mean_rep = np.dstack([gamma_fband_data_time_mean] * gamma_fband_data.shape[2])
            gamma_fband_data_norm = gamma_fband_data/gamma_fband_data_time_mean_rep

            # Append to list
            freq_band_means.append(gamma_fband_data_norm)

            pbar.update(1)

        # Take mean of all frequency bands
        gamma_trial_avg = np.array(freq_band_means).mean(axis=0)
        gamma_trial_avg = gamma_trial_avg.mean(axis=0)

        # Baseline correction
        if baseline_time != None:
            tvec = gamma_fband.times
            pre_stim_norm_time = baseline_time
            pre_stim_norm_time_idx = np.logical_and(tvec > pre_stim_norm_time[0], tvec < pre_stim_norm_time[1])
            gamma_trial_prestim = gamma_trial_avg[:, pre_stim_norm_time_idx].mean(axis=1)
            gamma_trial_avg_normalized = np.log(
                gamma_trial_avg / gamma_trial_prestim.reshape(gamma_trial_prestim.size, 1)) * 10
            #hfa_vals[evt] = gamma_trial_avg_normalized
            hfa_vals_evt = gamma_trial_avg_normalized
        else:
            #hfa_vals[evt] = gamma_trial_avg
            hfa_vals_evt = gamma_trial_avg

        hfa_evoked = EvokedArray(hfa_vals_evt,
                                 hfa_info,
                                 comment=evt,
                                 kind='average',
                                 tmin=evt_epoch.times[0],
                                 nave=evt_epoch.events.shape[0])
        hfa_evoked.set_montage(montage)
        del hfa_vals_evt
        hfa_list.append(hfa_evoked)

    # If only 1 event type, return the data from it. If multiple event types, keep the dictionary
    pbar.close()
    if len(hfa_list) == 1:
        return hfa_list[0]
    else:
        return hfa_list
    
    
def filter_hfa_continuous(data, hfa_fname, freq_range=[70, 170], freq_space='lin', n_freq_bins=10, convert_db=False, n_jobs=None, resample_fs=None):
    """Extract HFA from continuous data

    Args:
        data: LFP data as mne.io.Raw object 
        hfa_fname: name of output file
        freq_range: list of 2 elements defining range of HFA band (default: [70, 170])
        freq_space: choice of linear or logarithmic space of frequencies in filterbank ('lin', 'log') (default: 'lin')
        n_freq_bins: number of frequency bands in filterbank (default: 10)
        convert_db: convert HFA to decibel (default: False)
        n_jobs: Number of threads that can be used at once
        resample_fs: frequency to resample to after filtering (makes the HFA analysis more managable)

    Returns:  
        hfa_mne: HFA as mne.io.Raw object 

    Examples:
        hfa = filter_hfa_continuous(ecogReref, 'hfa.fif', [70, 180], 'log', 
                                    10, True, 16, 100)

    """ 
    
    if freq_space == 'log':
        f_bands = np.logspace(np.log10(freq_range[0]), np.log10(freq_range[1]), n_freq_bins+1)
    elif freq_space == 'lin':
        f_bands = np.linspace(freq_range[0], freq_range[1], n_freq_bins+1)
        
    filter_bank = list(zip(f_bands[:-1], f_bands[1:]))

    if resample_fs is not None:
        info_fs = resample_fs
    else:
        info_fs = data.info['sfreq']

    # Info object to use after
    ch_types = data.info.get_channel_types()
    hfa_info = create_info(data.ch_names, info_fs, ch_types)
    montage = data.get_montage()
    
    # Initiate a progress bar to track what's going on and how long it will take
    pbar = tqdm(total=len(filter_bank), ncols=100, desc='Running filter_hfa')

    # Loop through frequency bands by steps
    freq_band_means = []
    for fband in filter_bank:

        # Output is: CHANNELxTIME
        hfa_fband = data.copy().filter(fband[0], fband[1], n_jobs=n_jobs, verbose='ERROR').apply_hilbert(envelope=True, n_jobs=n_jobs)

        if resample_fs is not None:
            hfa_fband.resample(resample_fs)
           
        # Get data 
        hfa_fband_data = hfa_fband.get_data()
        
        # Optional conversion to decibel
        if convert_db:
            hfa_fband_data = 10*np.log10(hfa_fband_data)

        # Normalize it by dividing by mean over time
        hfa_fband_data_time_mean = np.abs(np.nanmean(hfa_fband_data,axis=1))
        hfa_fband_data_time_mean_rep = np.tile(hfa_fband_data_time_mean, 
                                               (hfa_fband_data.shape[1], 1)).T
        hfa_fband_data_norm = hfa_fband_data/hfa_fband_data_time_mean_rep

        # Append to list
        freq_band_means.append(hfa_fband_data_norm)

        pbar.update(1)
    
    # Take mean of all frequency bands
    freq_band_avg = np.nanmean(np.array(freq_band_means), axis=0)
    
    # Convert to MNE object
    if resample_fs is None:
        first_samp = 0
    else:
        first_samp = data.first_time*resample_fs
    hfa_mne = mne.io.RawArray(freq_band_avg, hfa_info, first_samp=first_samp)
    hfa_mne.set_montage(montage)
    
    # Save bad channels
    hfa_mne.info['bads'] = data.info['bads']
    
    # Save annotations
    hfa_mne.set_annotations(data.annotations)

    pbar.close()

    return hfa_mne
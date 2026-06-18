#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
import numpy as np
import pandas as pd
from scipy import interpolate
from mne.stats import permutation_cluster_1samp_test
from mne.time_frequency import tfr_morlet
from pycircstat2.hypothesis import rayleigh_test

# data is chanxtime and rayleigh z is calculated for each channel per frequency
def rayleigh_stat_fun(data):
    
    """Function to compute Rayleigh statistic on a 2D array
    
    Using the pycircstat2 package: https://github.com/circstat/pycircstat2
    
    Parameters
    ----------
    data : numpy array
        array of phase values [trial x time]

    Returns
    -------
    z_phase: numpy array 
        Rayleigh z-statistic across trials [time]

    """

    z_phase = np.empty(data.shape[1])
    for ii in range(data.shape[1]):
        z_phase[ii],_ = rayleigh_test(data[:,ii])
    return z_phase
        

def rayleigh_stat_mne(epochs, freq, n_cycles=5, crop=None, adjacency=None, 
                      tail=1, p_crit=0.05, n_permutations=100, n_jobs=1):
    
    """Compute ITC and test significance of time frequency phase clustering
    with cluster based statistics relying on the Rayleigh z-test
    
    Parameters
    ----------
    epcohs : instance of mne epochs object
        Data epoched to events of interest
    freq : list
        Lower and upper bound of frequency range of interest
    n_cycles : int (default n_cycles=5)
        Number of cycles for Morlet decomposition used to compute ITC and phase
    crops : list 
        Times in second to crop the epoch data
    adjacency : array
        Adjaceny between locations in the data, see MNE's 
        permutation_cluster_1samp_test() for details 
    tail : int (default tail=1)
        1: one-sided test above threshold, -1: one-sided test below threshold
        0 two-sided test
    p_crit : float (default=0.05)
        Critical p_value used to define threshold for clusters 
        Interpolated from a table and has to be in [0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001]
    n_permutation : int (default=100)
        Number of permutations for cluster statistics
    n_jobs : int (default=1)
        Number of parallel jobs
        
    Returns
    -------
    itc : instance of mne time-frequency object
        ITC across epochs
    phase : instance of mne time-frequency object
        Phase at all time-frequency points and epochs
    freqs : array
        Frequency points used in analysis
    z_obs : array [channels x freqs x times]
        Rayleigh z-values 
    clusters : list of arrays (each [channels x freqs x times]) for each cluster
        True/False masks for each cluster 
    cluster_p_values : list
        P-values for each cluster
    H0 : array [n_permutations]
        Max cluster level z-stats observed under permutation
    
    -------
    Examples:
        itc, phase, freqs, z_obs, clusters, cluster_p_values, H0 = mne_tfr.rayleigh_stat_mne(epochs['face'], 
                                                                                             freq=[5, 200], 
                                                                                             crop=[-0.1, 0.4], 
                                                                                             p_crit=0.05,
                                                                                             n_permutations=50,
                                                                                             n_jobs=16)
    """
    
    # Critical value for z-stat, maybe interpolate from this table
    # http://webspace.ship.edu/pgmarr/Geo441/Tables/Rayleighs%20z%20Table.pdf
    # From Zar 1981, Table B.32
    # 2D interpolation would be useful here, even better finding how this table
    # is computed
    rayleigh_z_values = pd.read_excel('rayleigh_z_values.xlsx')

    n = rayleigh_z_values.n.values
    n[-1] = sys.float_info.max          # Larget float, there might be a better way to define a value at infinite for interpolation

    z_vals = rayleigh_z_values.loc[:,p_crit].values

    f = interpolate.interp1d(n, np.squeeze(z_vals), fill_value='extrapolate')
    z_crit = f(epochs.__len__())
        
    # Set parameters for the tfr_morlet function
    nf = len(np.arange(freq[0], freq[1], 3))
    freqs = np.logspace(np.log10(freq[0]), np.log10(freq[1]), nf)
    
    # Get Phase of epochs
    phase = tfr_morlet(
        epochs,
        use_fft=True,
        average=False,
        decim=3,
        return_itc=False,
        output='phase',
        n_jobs=n_jobs,
        freqs=freqs,
        n_cycles=n_cycles)
    
    # Run itc (for itc trials MUST be averaged)
    _, itc = tfr_morlet(
        epochs['face'],
        use_fft=True,
        average=True,
        decim=3,
        return_itc=True,
        output='phase',
        n_jobs=n_jobs,
        freqs=freqs,
        n_cycles=n_cycles)
    
    itc.crop(crop[0],crop[1])

    # Make a copy of phase and crop it (reduces run time by only analyzing the window of interest)
    phase_crop = phase.copy()
    if crop is not None:
        phase_crop.crop(crop[0],crop[1])
    
    # Run the analysis
    z_obs, clusters, cluster_p_values, H0 = permutation_cluster_1samp_test(
        phase_crop.data,
        n_permutations=n_permutations,
        threshold=z_crit,
        stat_fun=rayleigh_stat_fun, #rayleigh_func_sample,
        tail=tail,
        n_jobs=n_jobs,
        adjacency=adjacency,
        out_type="mask",
        verbose=True)
    
    return itc, phase_crop, freqs, z_obs, clusters, cluster_p_values, H0
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  8 13:53:53 2023

@author: max

Example to test for significant clusters of phase based on Rayleigh statistics
"""

import sys
import numpy as np
import mne
import matplotlib.pyplot as plt

sys.path.append('/home/max/Documents/packages/EPIPE/Python')
from epipe import mne_tfr

#%% Parameters
crop = [-0.1, 0.4]
freq = [5, 200]
p_crit = 0.05
n_permutation = 50

# Load sample data
fname = '/home/max/Documents/Testing/sub-NS155_ses-implant02_task-visloc_acq-classic1_ref-avg_epo.fif.gz'
epochs = mne.read_epochs(fname, preload=True)

itc, phase, freqs, z_obs, clusters, cluster_p_values, H0 = mne_tfr.rayleigh_stat_mne(epochs['face'], 
                                                                                     freq=[5, 200], 
                                                                                     crop=[-0.1, 0.4], 
                                                                                     p_crit=0.05,
                                                                                     n_permutations=50,
                                                                                     n_jobs=16)

plt.figure()

# Discard z-values of non-significant time points
z_obs_plot = np.nan * np.ones_like(z_obs)
for c, p_val in zip(clusters, cluster_p_values):
    if p_val <= 0.05:
        z_obs_plot[c] = z_obs[c]

# Just plot one channel's data
ch_idx = np.where(np.in1d(phase.ch_names, 'LOm8'))[0][0]

vmax = np.max(np.abs(z_obs))
vmin = -vmax
plt.imshow(
    z_obs[ch_idx],
    cmap=plt.cm.gray,
    extent=[phase.times[0], phase.times[-1], freqs[0], freqs[-1]],
    aspect="auto",
    origin="lower",
    vmin=vmin,
    vmax=vmax,
)
plt.imshow(
    z_obs_plot[ch_idx],
    cmap=plt.cm.RdBu_r,
    extent=[phase.times[0], phase.times[-1], freqs[0], freqs[-1]],
    aspect="auto",
    origin="lower",
    vmin=vmin,
    vmax=vmax,
)
plt.colorbar()
plt.xlabel("Time (ms)")
plt.ylabel("Frequency (Hz)")
plt.title(f"Induced power ({phase.ch_names[ch_idx]})")

plt.show()

# Plot ITC
itc.plot([ch_idx], title='ITC')
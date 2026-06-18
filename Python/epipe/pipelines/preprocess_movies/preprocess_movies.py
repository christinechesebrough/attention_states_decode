# -*- coding: utf-8 -*-
"""
Created on Fri Jul 21 09:55:49 2023

@author: maxne
"""

#%% To-Do
# Need to label spikes and other artifacts in data

import os, sys
from itertools import compress
import numpy as np
import mne
from pynwb import NWBHDF5IO
import matplotlib

sys.path.append('/home/max/Documents/packages/EPIPE/Python')
from epipe import inspectNwb, nwb2mne, read_ielvis, reref_avg, reref_bipolar, filter_hfa_continuous

# After plotting raw channels and picking new ones to reject the kernel freezes
# One solution is disabling "Active support" for Matplotlib 
# (under tools -> preferences -> IPython console -> Graphics)
# And then manually setting the matplotlib backend, as described here:
# https://github.com/mne-tools/mne-python/issues/6528#issuecomment-892066104
matplotlib.use('Qt5Agg')

#%% Define directories and parameters
data_dir = '/media/max/Workspace/Data/Movies'
nwb_dir = ['Recordings', 'Neural']
prep_dir = ['Recordings', 'Neural_prep']
hfa_dir = ['Recordings', 'HFA']
et_dir = ['Recordings', 'Eyetracking']
anat_dir = 'Anatomy'

pat = 'NS127_02'

full_task_name = 'movies'
pipeline_name = 'preprocess_movies'
pipeline_version = 'v.0.0.0'

fs_dir = '{:s}/{:s}'.format(data_dir, anat_dir)

resample_fs = 600
notch_freqs = (60, 120, 180)

# Types of references to use in ieeg_analyses
# Must be a list containing at least one of the options: "avg", "bip"
ref_types = ['avg', 'bip']

n_jobs = 16

convert_db = True

# Frequency range
freq_range = [70, 170]
n_freq_bins = 10
freq_space = 'log'      # 'log', 'lin'

resample_bha_fs = 100

#%% Read data
nwb_dir = '{:s}/{:s}/{:s}/{:s}'.format(data_dir, nwb_dir[0], pat, nwb_dir[1])
sub_fsdir = '{:s}/{:s}'.format(fs_dir, pat)

nwb_files = os.listdir(nwb_dir)
idx_nwb = ['.nwb' in f for f in nwb_files]
nwb_files = list(compress(nwb_files, idx_nwb))

nwb_file = nwb_files[0]

nwb_fname = '{:s}/{:s}'.format(nwb_dir, nwb_file)

# NWB read
io = NWBHDF5IO(nwb_fname, mode='r', load_namespaces=True)
nwb = io.read()

# Get info on data in NWB file
nwbInfo = inspectNwb(nwb)
tsInfo = nwbInfo['timeseries']
elecTable = nwbInfo['elecs']

# Get ieeg data
if 'ieeg' in tsInfo['name'].to_list():
    ecogContainer = nwb.acquisition.get('ieeg')
    fs = ecogContainer.rate
    ecog = nwb2mne(ecogContainer,preload=False)

    # Get coordinates of each electrode that has
    ielvis_df = read_ielvis(sub_fsdir)
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

    # Create `montage` data structure as required by MNE
    montage = mne.channels.make_dig_montage(ch_pos=ch_coords, coord_frame='mri')
    montage.add_estimated_fiducials(pat, fs_dir)
    ecog.set_montage(montage)

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

#%% Pupil data
t_pupil = nwb.processing['eye_tracking']['pupils']['l_eye'].timestamps[:]
l_pupil = nwb.processing['eye_tracking']['pupils']['l_eye'].data[:]
r_pupil = nwb.processing['eye_tracking']['pupils']['r_eye'].data[:]

pupil = np.stack((l_pupil, r_pupil))

# Close NWB file
io.close()

#%% Preprocessing
sub_prep_dir = '{:s}/{:s}/{:s}/{:s}'.format(data_dir, prep_dir[0], pat, prep_dir[1])

if not os.path.exists(sub_prep_dir):
    os.makedirs(sub_prep_dir)
    
#region Notch filter, down sample, add/remove bad channels by inspecting raw trace, save
preproc_filename = '{:s}/{:s}'.format(sub_prep_dir, nwb_file.replace('.nwb', '_prep.fif'))

if not os.path.exists(preproc_filename):
    
    print('--->Applying notch filters and downsampling to %2.fHz' % resample_fs)
    
    # Copy the `ecog` variable and then resample and apply notch filter
    ecogPreproc = ecog.resample(resample_fs).notch_filter(notch_freqs, notch_widths=2)
    
    # Display the raw traces and mark bad channels
    nbadOrig = ecogPreproc.info['bads']
    fig = ecogPreproc.plot(show=True, block=True, remove_dc=True, duration=15.0, n_channels=16)
    
    # Save the current state of the data in the MNE format
                                          
    ecogPreproc.save(preproc_filename, 
                     fmt='single', overwrite=True)

else:
    ecogPreproc = mne.io.read_raw(preproc_filename)

# This loop runs all other steps on all types of references specified to use
for ref in ref_types:

    print('#' * 50)
    print('Beginning processing for data using %s reference' % ref)
    print('#' * 50)

    # What the preprocessed filename for this reference type should be
    preprocRerefFname = '{:s}/{:s}'.format(sub_prep_dir, 
                                           nwb_file.replace('.nwb', '_prep_ref_{:s}.fif'.format(ref)))

    # Check if the preprocessed file already exists so you don't have to redo rereferncing functions
    if os.path.isfile(preprocRerefFname):
        ecogReref = mne.io.read_raw_fif(preprocRerefFname, preload=True)
        if 'ecogPreproc' in locals():
            del ecogPreproc
    else:
        if 'ecogPreproc' not in locals():
            ecogPreproc = mne.io.read_raw_fif(preproc_filename, preload=True)

        if ref == 'avg':
            ecogReref = reref_avg(ecogPreproc)

        elif ref == 'bip':
            ecogReref = reref_bipolar(ecogPreproc)

        # Save the referenced data in MNE format
        ecogReref.save(preprocRerefFname,fmt='single',overwrite=True)
        del ecogPreproc
        
    #%% Filter for HFA
    sub_hfa_dir = '{:s}/{:s}/{:s}/{:s}'.format(data_dir, hfa_dir[0], pat, hfa_dir[1])
    hfa_fname = '{:s}/{:s}'.format(sub_hfa_dir,
                                   nwb_file.replace('.nwb', 
                                                    '_prep_ref_{:s}_hfa.fif'.format(ref)))

    if not os.path.exists(sub_hfa_dir):
        os.makedirs(sub_hfa_dir)
        
    if not os.path.exists(hfa_fname):
        
        # Compute HFA
        hfa_mne = filter_hfa_continuous(ecogReref, hfa_fname, freq_range, freq_space, n_freq_bins,
                                        convert_db, n_jobs, resample_bha_fs)
    
            
        # Cut by triggers
        hfa_data = hfa_mne.get_data()
        
        fs_hfa = hfa_mne.info['sfreq']
        times = np.arange(0, hfa_mne.times[-1] - (1/fs_hfa), 1/fs_hfa)
        
        ttl_id_start = np.where(ttl_id == 2)[0][0]
        ttl_id_end = np.where(ttl_id == 1)[0][1]
        
        sample_start = int(np.round(np.interp(ttls[ttl_id_start], times, 
                                              np.arange(0, len(times)))))
        
        sample_end = int(np.round(np.interp(ttls[ttl_id_end], times, 
                                            np.arange(0, len(times)))))
    
        hfa_data = hfa_data[:, sample_start:sample_end]
        
        # Convert to MNE object
        montage = hfa_mne.get_montage()
        hfa_mne = mne.io.RawArray(hfa_data, hfa_mne.info, first_samp=sample_start)
        hfa_mne.set_montage(montage)
        
        # Save the cut HFA data
        hfa_mne.save(hfa_fname,fmt='single',overwrite=True)
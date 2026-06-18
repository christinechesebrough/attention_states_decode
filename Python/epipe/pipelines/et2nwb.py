#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Adding Eye-Tracking Data to NWB Files
This project is in conjunction with Schroeder Lab (NKI) and the Parra Lab (CCNY)

Authors
Noah Markowitz
Gelana Tostaeva
Human Brain Mapping Lab
North Shore University Hospital
October 2021

CHANGELOG

December 16 2021
Ammended the method by which timestamps are constructed for the freeview task.
This new way preserves the time between timepoints and assures no overlap.
This is opposed to the prior method which aligned each trial of freeview to
a TTL timepoint. This caused difference in timepoints between eye-tracking
data ranging from 0.001ms-0.0006ms.

April 5 2022
Add support for TTLs being embedded in an analog channel for when data
recorded using Natus. Add `ana2dig()` and `_get_ttls()` functions.
"""

# TODO:
#   - Set so movie offset code can be T12, T13 or T14
#   - Add visual error checks and summaries
#       * Show which eyetracking and eeg TTL pulse do/do not have an corresponding TTL in the other
#           - Show EEG and Eyetracking codes on plots
#       * For analog TTLs (from Natus)
#           - Have additional error check that shows supposed TTL onsets got from `ana2dig()`
#             with analog timeseries overlayed. Use a while loop to continue updating until satisfied.
#           - For TTL correspondence plot (previous point) show timeseries, EEG TTLs as markers, and
#             eyetracker TTLs as vertical lines (using plt.axvline()) with green having a corresponding
#             TTL and red not having a corresponding

# %% Setup
# Import
from pynwb import NWBHDF5IO
from itertools import compress
import numpy as np
import h5py
from hdmf.backends.hdf5.h5_utils import H5DataIO
#from ndx_events import TTLs
from ndx_events.events import TTLs

from pynwb import TimeSeries
from pynwb.behavior import EyeTracking, PupilTracking
from pynwb.image import ImageSeries
import matplotlib.pyplot as plt
import sys
import os
import itertools
import pandas as pd
from scipy.interpolate import interp1d
from scipy.spatial.distance import pdist
from colorama import Back, Style

try:
    from mne.externals.pymatreader import read_mat
except:
    from pymatreader import read_mat


# Function to easily compress data
def wrap_data(data):
    return H5DataIO(
        data=data,
        compression='gzip',
        compression_opts=4,
        shuffle=True
    )

# Function to check if data is sorted
def issorted(data):
    if type(data) == 'list':
        data = np.array(data)

    return (np.arange(len(data)) == np.argsort(data)).all()

def ana2dig(arr,fs=None,thr=0.02,max_diff=10,rescale=[0,1],plot=False,return_time = False):
    from sklearn.preprocessing import minmax_scale
    arr = minmax_scale(arr,feature_range=rescale)
    
    if arr.ndim != 1:
        arr = np.amax(arr,axis=1)

    if fs != None:
        tvec = np.arange(0,len(arr),1) / fs
        max_diff_samples = int( np.ceil(max_diff*fs) )
    else:
        tvec = np.arange(0,len(arr),1)
        max_diff_samples = max_diff

    # Look at difference between points (derivative) for threshold
    arr_deriv = arr[1:] - arr[0:-1]
    arr_deriv_thr_idx = np.where(arr_deriv >= thr)[0]

    # Now check how far apart the points are.
    # If they're too close, they're part of a previous pulse
    arr_deriv_thr_idx_diff = np.diff(arr_deriv_thr_idx)
    arr_deriv_thr_idx_diff_small = arr_deriv_thr_idx_diff <= max_diff_samples
    #not_indiv_ttl = arr_deriv_thr_idx[np.where(arr_deriv_thr_idx_diff_small)[0] + 1]

    # Remove points that aren't TTLs
    indiv_ttls = np.delete(arr_deriv_thr_idx,np.where(arr_deriv_thr_idx_diff_small)[0] + 1)

    # Create a TTL
    is_ttl = np.zeros(( len(arr) ))
    is_ttl[indiv_ttls] = 1
    ttl_times = tvec[indiv_ttls]

    # Plot to check
    if plot:
        fig, ax = plt.subplots()
        ax.plot(tvec,arr)
        ax.plot(tvec[is_ttl.astype(bool)],arr[is_ttl.astype(bool)],'g*')

    if return_time:
        return is_ttl, ttl_times
    else:
        return is_ttl

# Get TTLs
def _get_ttls(nwbfile):

    # Find TTL container
    ttl_container_names = ['TTL','TTLS']
    acq_list = list(nwbfile.acquisition.keys())
    acq_list_upper = [c.upper() for c in nwbfile.acquisition.keys()]
    for acq in acq_list_upper:
        if acq in ttl_container_names:
            acqname = acq_list[ttl_container_names.index(acq)]
            return nwbfile.get_acquisition(acqname)

    return None

# Check TTLS: It's possible that some TTL pulses that occurred at the same time
# were treated as two different pulses
def check_ttls(nwb_fname):
    io_read = NWBHDF5IO(nwb_fname,mode='r',load_namespaces=True)
    nwb_read = io_read.read()

    # Load TTL data
    ttl_container = _get_ttls(nwb_read)

    if not isinstance(ttl_container,TimeSeries):
        # TTL dataframe
        ttls = {
            'data': ttl_container.timestamps[()],
            'code': []
        }
        for ii in ttl_container.data[()]:
            thisLabel = ttl_container.labels[ii]
            ttls['code'].append(thisLabel)

        ttls_df = pd.DataFrame(ttls)
        ttls_df['data'] = ttls_df['data'].round(3)

        # Check
        unique_times = np.unique(ttls_df['data'])
        num_unique = unique_times.shape[0]
        needs2change = ttls_df['data'].shape[0] != num_unique
    else:
        needs2change = False

    if needs2change:
        
        print('--->Before starting, the TTL container has to be adjusted')
        
        # Store the new TTLS
        new_ttls = {'time': [], 'stores': []}
        
        # Iterate through each unique time
        for t in unique_times:
            new_ttls['time'].append(t)
            codes_list = ttls_df.loc[ttls_df['data'] == t,'code'].tolist()
            
            # Concatenate codes
            new_code_list = []
            for l in codes_list:
                if '/' in l:
                    c = l.split('/')
                else:
                    c = [l]
                    
                new_code_list = new_code_list + c
                
            # Sort and join
            new_code_list.sort()
            new_code = '/'.join(new_code_list)
            new_ttls['stores'].append(new_code)
            

        df = pd.DataFrame(new_ttls)
        
        # Make codes for labels
        unique_ids = df['stores'].unique()
        label_vals = list(range(unique_ids.size))
        store_times = df['stores'].tolist()
        store_codes = dict(zip(unique_ids, label_vals))
        codes = []
        for cc in store_times:
            codes.append(store_codes[cc])
        
        # Create new TTLs container
        new_container = TTLs(
            name=ttl_container.name,
            description=ttl_container.description,
            timestamps=df['time'].to_numpy(),
            data=codes,
            labels=unique_ids)
        
        nwb_read.acquisition.pop('TTL')
        nwb_read.add_acquisition(new_container)
        
        # Write to file
        new_fname = nwb_fname[:-4] + '_new.nwb'
        with NWBHDF5IO(new_fname,mode='w') as export_io:
            export_io.export(src_io=io_read,nwbfile=nwb_read)
            
        # Rename and delete
        io_read.close()
        os.remove(nwb_fname)
        os.rename(new_fname,nwb_fname)
        
    else:
        io_read.close()

# Specs of the monitor
monitor_res = [1920,1080]
monitor_inches = 23
refresh_rate = 30
time_per_frame = 1/refresh_rate

# %% Eye-tracker CODES

# T255 is sent at the beginning and end of the experiment script
# T255 - Start/End of experiment

# Some earlier experiments require central fixation before movie start
# T101 - Start of fixation 
# T102 - End of fixation 

# Some recordings have triggers for specific movies 
# T241 - Despicable Me English
# T242 - Despicable Me Hungarian
# T243 - The Present
# T244 - Inscapes
# T255 - Monkey

# Freeview
# T7 - Image onset
# T9 - Button press

# Movies
# T11 - First sample of the movie
# T12/T13 - Just after the last movie of the frame was refreshed
# T55 - Pulse sent every 5sec to keep clocks synced

def run(task,nwb_fname,et_fname):
    
    # %% Check if TTLS are ok
    check_ttls(nwb_fname)
    
    # %% Define the task and make sure it's valid
    
    task_list = ['freeview', 'movie', 'fixation']
    task = task.lower()
    
    if task not in task_list:
        sys.exit('Please choose one of these for type of task: freeview, fixation, movie')
    
    # %% Load NWB and eye-tracking data
            
    # Import the file and get TTLs
    print(f'--->Reading NWB file:{nwb_fname}')
    io = NWBHDF5IO(nwb_fname,mode='r+',load_namespaces=True)
    nwb = io.read()
    ttl_container = _get_ttls(nwb)
    
   # if 'eye_tracking' in nwb.processing:
   #     io.close()
   #     return
    
    # Load eye-tracking data
    print(f'--->Reading eye-tracking mat file{et_fname}')
    eyeData_tmp = read_mat(et_fname)
    varname = [x for x in eyeData_tmp.keys() if not x.startswith('__')]
    eyeData = eyeData_tmp[varname[0]]
    del eyeData_tmp
    
    # Get movie file name
    if 'moviefile' in eyeData.keys():
        mov_file = eyeData['moviefile'].split('\\')[-1]
    
    # %% Load TTLs

    # Handle TTLs based on datatype
    # TTL pulses from NWB file
    if isinstance(ttl_container,TTLs):
        ttls_dict = {
            'eeg_ttl_time': ttl_container.timestamps[()],
            'eeg_ttl_lab':ttl_container.data[()],
            'eeg_code': []
        }
        for ii in ttl_container.data[()]:
            thisLabel = ttl_container.labels[ii]
            ttls_dict['eeg_code'].append(thisLabel)

    elif isinstance(ttl_container, TimeSeries):
        ttl_ts = ttl_container.data[()]
        ttl_fs = ttl_container.rate
        ttl_tvec = np.arange(0,len(ttl_ts),1) / ttl_fs
        is_ttl_onset, ttl_onsets = ana2dig(ttl_ts,fs=ttl_fs,thr=0.02,
                                           max_diff=0.01,plot=False,
                                           return_time = True)
        ttls_dict = {
            'eeg_ttl_time': ttl_onsets,
            'eeg_code': ['NA']*len(ttl_onsets)
        }
        

    # Get data from eye-tracking file
    timing_list = eyeData['timing']['ET_time']
    if isinstance(timing_list[0], list):
        et_codes = [x[0] for x in timing_list]
        et_ttl_codes = [x[1] for x in timing_list]
        ttls_dict['et_code'] = et_codes  # keep as list
        ttls_dict['et_ttl_time'] = np.array(et_ttl_codes).astype('float64')
    else: # for now (6/10/25) keep as py array instead of numpy array
        nTrigs = int( len(timing_list)/2 )
       # ttls_dict['et_code'] = np.array( timing_list[:nTrigs] )
       # ttls_dict['et_ttl_time'] = np.array( timing_list[nTrigs:] ).astype('float64')
            
        ttls_dict['et_code'] = timing_list[:nTrigs]  # keep as list
        ttls_dict['et_ttl_time'] = np.array(timing_list[nTrigs:]).astype('float64')

        
    #%% Estimate system time stamps if they were not recorded
    # (Interpolate from last trigger and correct for mean drift estimated from other recordings)
    if 'gaze_data' in eyeData.keys():
        
        if np.sum(eyeData['gaze_data']['dev_times'] 
                  - eyeData['gaze_data']['sys_times']) == 0:
            
            drift_estimate = 4.6148e-05
            
            dev_times = eyeData['gaze_data']['dev_times']
            T = dev_times[-1] - dev_times[0]
            
            dev_time_ttl = dev_times[-1]
            intercept_ttl = ttls_dict['et_ttl_time'][-1] - dev_time_ttl + T*drift_estimate
            
            eyeData['gaze_data']['sys_times'] = dev_times + intercept_ttl - ((dev_times - dev_times[0]) * drift_estimate)
                    
    #%% Handle subject-specifc discrepencies in number of TTLs
    
    # LH patients are missing triggers
    if nwb.subject.subject_id == 'sub-01' and task == 'movie':
    
        # Mostly 5s triggers are missing -> find these gaps
        ttl_diff = np.diff(ttls_dict['eeg_ttl_time'])
        idx_gap = ttl_diff > 10
        
        # Find how many triggers are missing
        t_gap = ttl_diff[idx_gap]
        n_ttl_gap = np.round(t_gap / 5) - 1
        
        fill_array = np.empty(len(n_ttl_gap), dtype=object)
        
        for n in range(len(n_ttl_gap)):
            fill_array[n] = np.nan * np.empty(int(n_ttl_gap[n]))
        
        eeg_ttl_filled = np.insert(ttls_dict['eeg_ttl_time'].astype(object), 
                                   np.where(idx_gap)[0] + 1, 
                                   fill_array)
        
        # Create an array
        flat_list = []
        
        for e in eeg_ttl_filled:
            if isinstance(e, np.ndarray):
                flat_list.extend(e.tolist())
            else:
                flat_list.append(e)
                
        # Overwrite dictionary entrance
        ttls_dict['eeg_ttl_time'] = np.array(flat_list)
        
        # Fill code entries
        ttl_code_edit = np.array(['DC1'] * len(ttls_dict['eeg_ttl_time']))    
        ttl_code_edit[np.invert(np.isnan(ttls_dict['eeg_ttl_time']))] = ttls_dict['eeg_code']
        
        ttls_dict['eeg_code'] = ttl_code_edit
        
    elif nwb.subject.subject_id in ['sub-01', 'sub-02'] and task == 'freeview':
        
        et_time = ttls_dict['et_ttl_time'] / 1e6
        eeg_time = ttls_dict['eeg_ttl_time']
        
        # First check if the first samples match
        fs_ieeg = nwb.acquisition['ieeg'].rate
        fs_eye = np.median([np.median(1/np.diff(T)) * 1e6 for T in eyeData['TS_fv']])
        
        fs_eye = 1 / np.median(np.diff(eyeData['vbl_aggr'] * 1e6))
        #fs_eye = 1 / np.median(np.diff(eyeData['vbl_aggr']))

        
        if np.abs((et_time[1]-et_time[0]) - (eeg_time[1]-eeg_time[0])) > 1/fs_ieeg:
            
            print(Back.RED + 'Cannot align based on first trigger')
            print(Style.RESET_ALL)
            
            return
        
        # Create a filled vector
        eeg_time_filled = np.zeros(len(et_time)) * np.nan
        eeg_time_filled[0] = eeg_time[0]
        
        # Counter for EEG time
        i_et = 1
        i_eeg = 1
        
        # In a loop step through each eyetracking trigger and check if the
        # time to the next iEEG trigger matches, if not step through the 
        # following iEEG triggers until a match is found
        
        for t in range(1, len(et_time)):
            
            eeg_time_now = eeg_time[i_eeg]
            eeg_time_pre = eeg_time[i_eeg-1]
            
            et_time_pre = et_time[i_et-1]
            
            dt_et = et_time[t] - et_time_pre
            dt_eeg = eeg_time_now - eeg_time_pre

            if np.abs(dt_et - dt_eeg) < 1/fs_eye:
                
                eeg_time_filled[t] = eeg_time_now
                i_eeg += 1
                i_et = t+1
                
        ttls_dict['eeg_ttl_time'] = eeg_time_filled
        ttls_dict['eeg_code'] = ['DC1'] * len(eeg_time_filled)
#%%
        
    # Some blocks have a different amount of TTLs between the ET and EEG data
    print(f"Number of TTLs in EEG: {len(ttls_dict['eeg_ttl_time'])}")
    print(f"Number of TTLs in Eye Tracker: {len(ttls_dict['et_ttl_time'])}")
    if len(ttls_dict['eeg_ttl_time']) != len(ttls_dict['et_ttl_time']):
        print(Back.YELLOW + 'Number of triggers in EEG and eyetracking don\'t match!')
        print(Style.RESET_ALL)

    #%% 
    #Get rid of T8 if they weren't part of the TTLs (check first to see if there are any T8 events)
    if task == 'freeview' and (len(ttls_dict['et_code']) != len(ttls_dict['eeg_code'])):

        # Find indices of entries NOT labeled 'T8'
        not_t8_idx = [i for i, code in enumerate(ttls_dict['et_code']) if code != 'T8']
        
        num_t8 = len(ttls_dict['et_code']) - len(not_t8_idx)
        print(f"Found {num_t8} 'T8' events in et_code.")
    
        # Apply removal only if any T8s are found
        if num_t8 > 0:
            ttls_dict['et_code'] = ttls_dict['et_code'][not_t8_idx]
            ttls_dict['et_ttl_time'] = ttls_dict['et_ttl_time'][not_t8_idx]
            ttls_dict['eeg_ttl_lab'] = ttls_dict['eeg_ttl_lab'][not_t8_idx]

            # Check that lengths now match
            assert len(ttls_dict['et_code']) == len(ttls_dict['eeg_code']), (
                f"After removing T8s, et_code length ({len(ttls_dict['et_code'])}) "
                f"still does not match eeg_code length ({len(ttls_dict['eeg_code'])})."
            )
            print("Removed T8s successfully and lengths now match.")
        else:
            # No T8s found – log warning and assert to prevent silent errors
            print("WARNING: No 'T8' events found in et_code.")

    #%%
    if os.path.basename(et_fname) == 'Eye_fv_subNS174_03.mat':

        # Remove the event at index 22
        ttls_dict['eeg_ttl_time'] = np.delete(ttls_dict['eeg_ttl_time'], 6)
        ttls_dict['eeg_code'] = np.delete(ttls_dict['eeg_code'], 6)
        ttls_dict['eeg_ttl_lab'] = np.delete(ttls_dict['eeg_ttl_lab'], 6)
        print(f"Number of TTLs in EEG: {len(ttls_dict['eeg_ttl_time'])}")
        print(f"Number of TTLs in Eye Tracker: {len(ttls_dict['et_ttl_time'])}")

    
    #%% Check these
    # Eliminate first 3 TTLs for this block
    if os.path.basename(et_fname) == 'Eye_movie_sublij134_Monkey2avi_rep1.mat':
        ttls_dict['eeg_code'] = ttls_dict['eeg_code'][3:]
        ttls_dict['eeg_ttl_time'] = ttls_dict['eeg_ttl_time'][3:]

    # Eliminate last triggers for this block, where iEEG was cut short
    if os.path.basename(et_fname) == 'Eye_movie_subns151_The_Presentmp4_rep1.mat':
        n_trigger = len(np.diff(ttls_dict['eeg_ttl_time']))
        ttls_dict['et_ttl_time'] = ttls_dict['et_ttl_time'][:n_trigger+1]
        ttls_dict['et_code'] = ttls_dict['et_code'][:n_trigger+1]
        
    #%%
    #if nwb.subject.subject_id == 'sub-35' and 'Despicable_Me_English.mp4' in nwb.notes:
    
    placeholders = False    
    
    if len(ttls_dict['eeg_ttl_time']) != len(ttls_dict['et_ttl_time']):    
       
        subject_id = nwb.subject.subject_id
        remove_dups = False  # default
        
        if subject_id.startswith('sub-'):
            # Extract the numeric part and convert to int
            sub_number = int(subject_id.replace('sub-', ''))
            if sub_number > 31:
                remove_dups = True
        
        if remove_dups:
            # Find duplicates
            eeg_ttl_times = ttls_dict['eeg_ttl_time']
            is_duplicate = []
            keep_mask = []
            is_duplicate = np.diff(eeg_ttl_times) < 0.003
            
            # Pad to match array length
            is_duplicate = np.hstack((False, is_duplicate))
            #is_duplicate = np.hstack((is_duplicate, False))  # instead of padding at the front

            # Create full mask of values to keep
            keep_mask = np.ones(len(eeg_ttl_times), dtype=bool)

            if 'Despicable_Me_English.mp4' in nwb.notes:
                
                dup_idx = 0
                keep_mask[dup_idx:] = np.invert(is_duplicate[dup_idx:])
                
                #if sub_number == 32:
                #    keep_mask[-1]=True
                #    keep_mask[7] = True
                
                if os.path.basename(nwb_fname) == 'sub-32_ses-02_task-despicable_me_english_run-01_ieeg.nwb':
                    keep_mask[0]=False
                    keep_mask[2]=False
                    keep_mask[4]=False
                    keep_mask[6]=False
                
                if sub_number >32 and sub_number < 39:
                    placeholders = True

                # Apply to both times and codes
                ttls_dict['eeg_ttl_time'] = eeg_ttl_times[keep_mask]
                ttls_dict['eeg_code'] = list(compress(ttls_dict['eeg_code'], keep_mask))
                ttls_dict['eeg_ttl_lab']=ttls_dict['eeg_ttl_lab'][keep_mask]
                
                
            elif 'Partysaurus_Rex.mp4' in nwb.notes:
                placeholders = False
                #dup_idx = 0
                #keep_mask[dup_idx:] = np.invert(is_duplicate[dup_idx:])

                #Remove et_codes == 1
                eeg_codes = np.array(ttls_dict['eeg_ttl_lab'])  # ensure array for masking
                eeg_times = np.array(ttls_dict['eeg_ttl_time'])
                eeg_keep_mask = eeg_codes != 1
                eeg_keep_mask[-2]=False
                ttls_dict['eeg_ttl_lab'] = ttls_dict['eeg_ttl_lab'][eeg_keep_mask].tolist()
                ttls_dict['eeg_ttl_time']=ttls_dict['eeg_ttl_time'][eeg_keep_mask].tolist()
                ttls_dict['eeg_code']=ttls_dict['eeg_code'][eeg_keep_mask].tolist()
            
            else:             
                dup_idx = 0
                keep_mask[dup_idx:] = np.invert(is_duplicate[dup_idx:])
                is_duplicate = np.hstack((False, is_duplicate))
                
                # Apply to both times and codes
                ttls_dict['eeg_ttl_time'] = eeg_ttl_times[keep_mask]
                ttls_dict['eeg_code'] = list(compress(ttls_dict['eeg_code'], keep_mask))
                ttls_dict['eeg_ttl_lab']=ttls_dict['eeg_ttl_lab'][keep_mask]
                
                
            # after deleting dups
            print(f"Number of TTLs in EEG after deleting duplicates: {len(ttls_dict['eeg_ttl_time'])}")
            print(f"Number of TTLs in Eye Tracker after deleting duplicates: {len(ttls_dict['et_ttl_time'])}")
           #%%     
            if placeholders == True:
                ttls_dict['eeg_ttl_time'] = np.insert(ttls_dict['eeg_ttl_time'], 4, np.nan)
                ttls_dict['eeg_code'] = np.insert(ttls_dict['eeg_code'], 4, 'PrtA')        # Placeholder code
                ttls_dict['eeg_ttl_lab']=np.insert(ttls_dict['eeg_ttl_lab'],4,'0')
                
                ttls_dict['eeg_ttl_time'] = np.hstack((ttls_dict['eeg_ttl_time'], np.nan))
                ttls_dict['eeg_code'] = np.hstack((ttls_dict['eeg_code'], 'PrtA'))
                ttls_dict['eeg_ttl_lab']=np.hstack((ttls_dict['eeg_ttl_lab'],'0'))

                                
            # after deleting dups
            print(f"Number of TTLs in EEG after deleting duplicates: {len(ttls_dict['eeg_ttl_time'])}")
            print(f"Number of TTLs in Eye Tracker after deleting duplicates: {len(ttls_dict['et_ttl_time'])}")
        
            
#%%
    if os.path.basename(et_fname) == 'Eye_movie_sublh010_Despicable_Me_Hungarianmp4_rep1.mat':
        ttls_dict['et_ttl_time'] = ttls_dict['et_ttl_time'][:-1]
        ttls_dict['et_code'] = ttls_dict['et_code'][:-1]   
        
    if os.path.basename(et_fname) == 'Eye_movie_sublh010_The_Presentmp4_rep1.mat':
        ttls_dict['et_ttl_time'] = ttls_dict['et_ttl_time'][1:]
        ttls_dict['et_code'] = ttls_dict['et_code'][1:] 
        
    if os.path.basename(et_fname) == 'Eye_movie_sublh010_The_Presentmp4_rep2.mat':
        ttls_dict['eeg_ttl_time'] = np.insert(ttls_dict['eeg_ttl_time'], 2, np.nan)
        ttls_dict['eeg_code'] = np.insert(ttls_dict['eeg_code'], 2, 'DC1')
    
    if os.path.basename(et_fname) == 'Eye_movie_sublh010_Monkey1avi_rep1.mat':
        ttls_dict['eeg_ttl_time'] = np.insert(ttls_dict['eeg_ttl_time'], [2,2], np.nan)
        ttls_dict['eeg_code'] = np.insert(ttls_dict['eeg_code'], [2,2], 'DC1')
        
    if os.path.basename(et_fname) == 'Eye_movie_sublh010_Monkey2avi_rep1.mat':
        ttls_dict['eeg_ttl_time'] = np.insert(ttls_dict['eeg_ttl_time'], 0, np.nan)
        ttls_dict['eeg_code'] = np.insert(ttls_dict['eeg_code'], 0, 'DC1')
        
    if os.path.basename(et_fname) == 'Eye_movie_sublh010_Monkey5avi_rep1.mat':
        ttls_dict['eeg_ttl_time'] = np.insert(ttls_dict['eeg_ttl_time'], 3, np.nan)
        ttls_dict['eeg_code'] = np.insert(ttls_dict['eeg_code'], 3, 'DC1')
        
    if os.path.basename(et_fname) == 'Eye_movie_sublh010_PartySaurusRexmov_rep1.mat':
        ttls_dict['eeg_ttl_time'] = np.insert(ttls_dict['eeg_ttl_time'], 0, np.nan)
        ttls_dict['eeg_code'] = np.insert(ttls_dict['eeg_code'], 0, 'DC1')
        
    if os.path.basename(et_fname) == 'Eye_fv_subNS128_02.mat':
        ttls_dict['eeg_ttl_time'] = ttls_dict['eeg_ttl_time'][2:]
        ttls_dict['eeg_code'] = ttls_dict['eeg_code'][2:]
    
    if os.path.basename(et_fname) == 'Eye_movie_subNS174_03_Despicable_Me_Englishmp4_rep1.mat':
        # Remove the event at index 22
        ttls_dict['eeg_ttl_time'] = np.delete(ttls_dict['eeg_ttl_time'], 22)
        ttls_dict['eeg_code'] = np.delete(ttls_dict['eeg_code'], 22)
        ttls_dict['eeg_ttl_lab'] = np.delete(ttls_dict['eeg_ttl_lab'], 22)
    
    
    #%% Ajust missing triggers algorithmically 
    # By brute force trial and error -> try to delete any combination of extra triggers
    # from the longer array and see which one matches best
    # This works for a few missing triggers, for anything over 4 or 5 this gets too memory intensive
    # Some of these cases are defined above 
    
    # Stop if the difference is too big and add manual correction above
    n_diff = len(ttls_dict['et_ttl_time']) - len(ttls_dict['eeg_ttl_time'])
    
    if abs(n_diff) > 4:
        
        print(Back.RED + 'Number of triggers in EEG and ET data is off by 5 or more!')
        print(Style.RESET_ALL)
        
        io.close()
        return
    
    # Check which one is longer
    if len(ttls_dict['et_ttl_time']) > len(ttls_dict['eeg_ttl_time']):     
        ttl_long = ttls_dict['et_ttl_time']
        ttl_reference = ttls_dict['eeg_ttl_time']
    elif len(ttls_dict['eeg_ttl_time']) > len(ttls_dict['et_ttl_time']):     
        ttl_long = ttls_dict['eeg_ttl_time']
        ttl_reference = ttls_dict['et_ttl_time']

    if len(ttls_dict['et_ttl_time']) != len(ttls_dict['eeg_ttl_time']):
        
        # Find all combinations of indices of triggers to delete
        samples = list(itertools.combinations(np.arange(len(ttl_long)), 
                                              len(ttl_long)-len(ttl_reference)))
        
        dist = np.empty(len(samples))
        
        # Iterate through the list, delete the triggers and measure the match 
        # of the triggers by the euclidian distance between the time between 
        # all triggers
        for i in range(len(samples)):
            
            ttl_cut_test = ttl_long
            ttl_cut_test = np.delete(ttl_cut_test, samples[i])
            
            dist[i] = pdist(np.vstack((np.diff(ttl_cut_test)/1e6, 
                                       np.diff(ttl_reference))))
            
        # Find minimum distance, i.e. the triggers that match best when deleted
        idx_delete = np.argmin(dist)
    
    # Delete these triggers from the corresponding array 
    if len(ttls_dict['et_ttl_time']) > len(ttls_dict['eeg_ttl_time']):     
        ttls_dict['et_ttl_time'] = np.delete(ttls_dict['et_ttl_time'], samples[idx_delete])
        ttls_dict['et_code'] = np.delete(ttls_dict['et_code'], samples[idx_delete])
    elif len(ttls_dict['eeg_ttl_time']) > len(ttls_dict['et_ttl_time']):     
        ttls_dict['eeg_ttl_time'] = np.delete(ttls_dict['eeg_ttl_time'], samples[idx_delete])
        ttls_dict['eeg_code'] = np.delete(ttls_dict['eeg_code'], samples[idx_delete])
             
    # %% Create DataFrame for TTLs
    
    # Construct dataframe to easily compare times
    ttls_df = pd.DataFrame(ttls_dict)

    #%% Check timing of TTLs
    
    print("Creating figure...")

    fig_file = nwb_fname.replace('nwb/', 'nwb_conversion/alignment/')
    fig_file = fig_file.replace('_ieeg.nwb', '_trigger_alignment.png')
    
    fig_dir = os.path.dirname(fig_file)
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)
        
    # Time difference
    t_diff_interval = 1e3*(np.diff(ttls_df.eeg_ttl_time) - np.diff(ttls_df.et_ttl_time/1e6))
    
    plt.figure(figsize=[15,5])
    
    plt.subplot(2,1,1)
    
    plt.plot(np.diff(ttls_df.eeg_ttl_time), '*-')
    plt.plot(np.diff(ttls_df.et_ttl_time/1e6), '*-')
    
    plt.legend(['iEEG','ET'])#'iEEG', 
    
    plt.ylabel('Trigger interval [s]')
    plt.grid()
    
    plt.title(os.path.split(et_fname)[1])
    
    plt.subplot(2,1,2)
    
    plt.plot(t_diff_interval, 'g*-')
    
    plt.xlabel('Trigger interval')
    plt.ylabel('iEEG-ET interval diff [ms]')
    
    if task == 'movie':
        fs_eye = eyeData['setup']['currentFrameRate']
    elif task == 'freeview':
        fs_eye = np.median([np.median(1/np.diff(T)) * 1e6 for T in eyeData['TS_fv']])
    elif task == 'fixation':
        fs_eye = np.median(1/np.diff(eyeData['TS_fv'])) * 1e6
    
    fs_ieeg = nwb.acquisition['ieeg'].rate
    
    plt.plot([0, len(ttls_df.eeg_ttl_time)-1], 1e3/fs_eye*np.array([1,1]), 'r--', label='_nolegend_')
    plt.plot([0, len(ttls_df.eeg_ttl_time)-1], 1e3/fs_eye*np.array([-1,-1]), 'r--')
    
    plt.plot([0, len(ttls_df.eeg_ttl_time)-1], 1e3/fs_ieeg*np.array([1,1]), 'k--', label='_nolegend_')
    plt.plot([0, len(ttls_df.eeg_ttl_time)-1], 1e3/fs_ieeg*np.array([-1,-1]), 'k--')
    
    plt.legend(['interval', 'ET sample', 'iEEG sample'])
    
    plt.grid()
    
    plt.savefig(fig_file, dpi=300)
       
    # assert np.sum(t_diff_interval < 1) == len(t_diff_interval), 'Jitter of interval between triggers is > 1ms!'
    
    # %% BUILD Get initial events
    
    print('--->Indexing event labels')
    
    # TODO
    if task == 'fixation':
    
        # Set variables
        eyeRight = eyeData['REC_fv']
        eyeLeft = eyeData['LEC_fv']
        eyeTime = eyeData['TS_fv']
        et_fs = eyeData['setup']['fixDur']
    
        # Get trigger times
        calibration_idx = np.where(ttls_df['et_code'] == 'T255')[0][-2:]
        ttls_df['events'] = ['onset','offset']
        
        device_description = 'Tobii TX300 eye-tracker with built-in monitor {}inch and {}x{} resolution'
        
    # TODO
    elif task == 'freeview':
        
        # Set variables
        eyeRight = eyeData['REC_fv']
        eyeLeft = eyeData['LEC_fv']
        eyeTime = eyeData['TS_fv']
        ntrials = len(eyeRight)
        stimulus_time = 6
        et_fs = int(np.round(1 / (np.median([np.median(np.diff(et)) for et in eyeTime]) / 1e6)))
        
        # Set events and get rid of anything that's not an image event
        image_events_idx = ttls_df['et_code'] == 'T7'
        ttls_df = ttls_df[image_events_idx]
        ttls_df = ttls_df.reset_index(drop=True)
        ttls_df['events'] = ['image_onset'] * ntrials
        
        # Load images and image variables
        imgIDs = eyeData['trlmat_fv_final'][1,:].astype(int)
        imgRandomization = eyeData['trlmat_fv_final'][0,:]
    
        # Create an array that gives the precise order in which pictures were presented
        imgOrder = np.argsort(imgRandomization)
        imgsSorted = imgIDs[imgOrder]
        stim_labels = ['pic' + str(ii).zfill(3) for ii in imgsSorted]
        
        ntrials = ttls_df.shape[0]
        ttls_df['stim'] = stim_labels[0:ntrials]
        
        device_description = 'Tobii TX300 eye-tracker with built-in monitor {}inch and {}x{} resolution'
        
    elif task == 'movie':
        
        # Set variables
        if ['gaze_data' in eyeData.keys()][0]:
            
            # Convert validity to match old eyetracker as close as possible
            val_right = eyeData['gaze_data']['RightEye']['GazePoint']['Validity']
            val_left = eyeData['gaze_data']['LeftEye']['GazePoint']['Validity']
            
            val_left_only = np.logical_and(val_left == 1, val_right == 0)
            val_right_only = np.logical_and(val_right == 1, val_left == 0)
            val_none = np.logical_and(val_right == 0, val_left == 0)
            
            val_right = np.zeros((len(val_right), 1))
            val_right[val_right_only] = 1
            val_right[val_left_only] = 3
            val_right[val_none] = 4
            
            val_left = np.zeros((len(val_left), 1))
            val_left[val_left_only] = 1
            val_left[val_right_only] = 3
            val_left[val_none] = 4
            
            # Create an array as for the old eyetracker
            eyeRight = np.concatenate((
                eyeData['gaze_data']['RightEye']['GazeOrigin']['InUserCoordinateSystem'],
                eyeData['gaze_data']['RightEye']['GazeOrigin']['InTrackBoxCoordinateSystem'],
                eyeData['gaze_data']['RightEye']['GazePoint']['OnDisplayArea'],
                eyeData['gaze_data']['RightEye']['GazePoint']['InUserCoordinateSystem'],
                np.expand_dims(eyeData['gaze_data']['RightEye']['Pupil']['Diameter'], axis=1),
                val_right),
                axis=1)
            
            eyeLeft = np.concatenate((
                eyeData['gaze_data']['LeftEye']['GazeOrigin']['InUserCoordinateSystem'],
                eyeData['gaze_data']['LeftEye']['GazeOrigin']['InTrackBoxCoordinateSystem'],
                eyeData['gaze_data']['LeftEye']['GazePoint']['OnDisplayArea'],
                eyeData['gaze_data']['LeftEye']['GazePoint']['InUserCoordinateSystem'],
                np.expand_dims(eyeData['gaze_data']['LeftEye']['Pupil']['Diameter'], axis=1),
                val_left),
                axis=1)
            
            eyeTime = eyeData['gaze_data']['sys_times']         
            et_fs = eyeData['setup']['currentFrameRate']
            
            # Get time when each frame of the movie was presented
            sdk_time = eyeData['gaze_data']['sys_times']
            frame_times = eyeData['vbl_aggr'] * 1e6
            frame_times_sec = eyeData['vbl_aggr'] - eyeData['vbl_aggr'][0]

            # Get frame samples
            f_frame = interp1d(sdk_time, np.arange(len(sdk_time)), fill_value='extrapolate')
            frame_sample = np.round(f_frame(frame_times)).astype(int)
            
            device_description = 'Tobii Spectrum eye-tracker with built-in monitor {}inch and {}x{} resolution'
            
        else:
            
            eyeRight = eyeData['rightEye']
            eyeLeft = eyeData['leftEye']
            eyeTime = eyeData['timeStamp']
            et_fs = eyeData['setup']['currentFrameRate']
            
            # Get SDK time
            ttl_sdk = eyeData['timing']['SDK_time']  
            if type(ttl_sdk[0]) is str:
                idx_t = [type(e) is int for e in eyeData['timing']['SDK_time']]
                ttl_sdk = np.array(list(compress(ttl_sdk, idx_t)))
            elif type(ttl_sdk[0]) is list:
                ttl_sdk = [e[1] for e in ttl_sdk]
                
            ttl_et = eyeData['timing']['ET_time']   
            if type(ttl_et[0]) is str:
                idx_t = [type(e) is int for e in eyeData['timing']['ET_time']]
                ttl_et = np.array(list(compress(ttl_et, idx_t)))
            elif type(ttl_et[0]) is list:
                ttl_et = [e[1] for e in ttl_et]
                
            f_ttl = interp1d(ttl_et, ttl_sdk, fill_value='extrapolate')
            sdk_time = f_ttl(eyeTime)
            
            # Get time when each frame of the movie was presented
            frame_times = eyeData['vbl_aggr'] * 1e6
            frame_times_sec = (eyeData['vbl_aggr'] - eyeData['vbl_aggr'][0])

            # Get frame samples
            f_frame = interp1d(sdk_time, np.arange(len(sdk_time)), fill_value='extrapolate')
            frame_sample = np.round(f_frame(frame_times)).astype(int)
            
            device_description = 'Tobii TX300 eye-tracker with built-in monitor {}inch and {}x{} resolution'
            
            
         #%%   
        # Movie identity (inconsistent across patients and not needed)
        movie_id_codes = ['T255', 'T241', 'T242','T243','T244']
        movie_id_idx = np.where(np.in1d(ttls_df['et_code'], movie_id_codes))[0][:-1]
        
        # Pre movie fixation
        fix_start_idx = np.where(ttls_df['et_code'] == 'T101')[0]
        fix_end_idx = np.where(ttls_df['et_code'] == 'T102')[0]
        
        if fix_end_idx.size == 0:
            fix_end_idx = np.where(ttls_df['et_code'] == 'T103')[0]
        
        # For earlier subjects the movie onset code is different
        movie_onset_code = 'T11'
        
        alignment_pulses_idx = np.where(ttls_df['et_code'] == 'T55')[0]
        movie_onset_idx = np.where(ttls_df['et_code'] == movie_onset_code)[0]
        movie_offset_idx = np.where(ttls_df['et_code'] == 'T13')[0]
        
        # Movie offset may be coded as T12
        if movie_offset_idx.size == 0:
            movie_offset_idx = np.where(ttls_df['et_code'] == 'T12')[0]

        if movie_offset_idx.size == 0:
            movie_offset_idx = np.where(ttls_df['et_code'] == 'T14')[0]
            
        # Experiment end
        t255_idx = np.where(ttls_df['et_code'] == 'T255')[0]
        exp_end_idx = t255_idx[t255_idx > movie_onset_idx]
    
        # Add events to the dataframe
        ttls_df['events'] = 'NA'
        for i in range(len(movie_id_idx)):
            ttls_df.loc[movie_id_idx[i],['events']] = 'movie_id' 
        if len(fix_start_idx) == 1:
            ttls_df.loc[fix_start_idx,['events']] = 'fixation_start'
        if len(fix_end_idx) == 1:
            ttls_df.loc[fix_end_idx,['events']] = 'fixation_end'
        ttls_df.loc[alignment_pulses_idx,['events']] = 'alignment'
        ttls_df.loc[movie_onset_idx,['events']] = 'movie_onset'
        ttls_df.loc[movie_offset_idx,['events']] = 'movie_offset'
        ttls_df.loc[exp_end_idx,['events']] = 'eyetracker_callback'
    
        # Drop NA events
        ttls_df = ttls_df[ttls_df['events'] != 'NA']
        ttls_df = ttls_df.reset_index(drop=True)
        
        # Remove triggers outside of time axis 
        ttls_df = ttls_df[ttls_df['et_ttl_time'] < eyeTime[-1]]
        
    # Remove variables from memory
    #del eyeData
    
    # %% BUILD Find nearest sample points and index specific events
     
    # For freeview, this step will occur in the trial loop
    if task != 'freeview':
        
        f = interp1d(eyeTime, np.arange(len(eyeTime)), fill_value='extrapolate')
        trg_sample = np.round(f(ttls_df['et_ttl_time'])).astype(int)   
        
        ttls_df['et_nearest_sample_index'] = trg_sample
        ttls_df['et_nearest_timepoint'] = eyeTime[trg_sample]
        ttls_df['et_time_diff'] = (ttls_df['et_ttl_time'] - ttls_df['et_nearest_timepoint']) / 1e6
        
    # %% BUILD Get more events
    
    if task == 'movie':
        
        #############################################################
        # Some important events
        #############################################################
        
        # Number of eye-tracker samples
        nsamples = len(eyeTime)
    
        # Important events
        movie_onset_row = np.where(ttls_df['events'] == 'movie_onset')[0][0]
        
        if len(np.where(ttls_df['events'] == 'movie_offset')[0]) != 0:
            movie_offset_row = np.where(ttls_df['events'] == 'movie_offset')[0][0]
    
        #############################################################
        # Dictionary of some movie info
        #############################################################
        if 'movie_offset_row' in locals():
            
            movie_info = {
                'orig_fs': et_fs,
                'adjusted_fs': 1/(np.diff( eyeTime/10**6 ).mean()),
                'dur_et_sample_diff': (ttls_df.loc[movie_offset_row,'et_nearest_timepoint'] - ttls_df.loc[movie_onset_row,'et_nearest_timepoint'])/10**6,
                'dur_et_ttl_diff': (ttls_df.loc[movie_offset_row,'et_ttl_time'] - ttls_df.loc[movie_onset_row,'et_ttl_time'])/10**6,
                'dur_orig_fs': nsamples/et_fs,
                'dur_adj_fs': nsamples/(1/(np.diff( eyeTime/10**6 ).mean())),
                'dur_eeg_ttl_diff': (ttls_df.loc[movie_offset_row,'eeg_ttl_time'] - ttls_df.loc[movie_onset_row,'eeg_ttl_time']),
                'samples_actual': nsamples
                }
            movie_info['samples_et_fs_adj'] = movie_info['dur_et_sample_diff'] * movie_info['adjusted_fs']
            movie_info['samples_et_fs_orig'] = movie_info['dur_et_sample_diff'] * movie_info['orig_fs']
            movie_info['samples_et_ttl_fs_adj'] = movie_info['dur_et_ttl_diff'] * movie_info['adjusted_fs']
            movie_info['samples_et_ttl_fs_orig'] = movie_info['dur_et_ttl_diff'] * movie_info['orig_fs']
            movie_info['samples_et_sample_diff'] = ttls_df['et_nearest_sample_index'][movie_offset_row] - ttls_df['et_nearest_sample_index'][movie_onset_row]
        
        else:
            
            movie_info = {
                'orig_fs': et_fs,
                'adjusted_fs': 1/(np.diff( eyeTime/10**6 ).mean()),
                'dur_et_sample_diff': np.nan,
                'dur_et_ttl_diff': np.nan,
                'dur_orig_fs': nsamples/et_fs,
                'dur_adj_fs': nsamples/(1/(np.diff( eyeTime/10**6 ).mean())),
                'dur_eeg_ttl_diff': np.nan,
                'samples_actual': nsamples
                }
            movie_info['samples_et_fs_adj'] = movie_info['dur_et_sample_diff'] * movie_info['adjusted_fs']
            movie_info['samples_et_fs_orig'] = movie_info['dur_et_sample_diff'] * movie_info['orig_fs']
            movie_info['samples_et_ttl_fs_adj'] = movie_info['dur_et_ttl_diff'] * movie_info['adjusted_fs']
            movie_info['samples_et_ttl_fs_orig'] = movie_info['dur_et_ttl_diff'] * movie_info['orig_fs']
            movie_info['samples_et_sample_diff'] = np.nan
            
    
        #############################################################
        # Align events and create time-vector for eye-tracker data
        #############################################################
        
        # Interpolate the time axes to account for drift
        et_time_interp = ttls_df.et_ttl_time.values
        eeg_time_interp = ttls_df.eeg_ttl_time.values
        
        idx_good = np.invert(np.isnan(eeg_time_interp))
        
        et_time_interp = et_time_interp[idx_good]
        eeg_time_interp = eeg_time_interp[idx_good]
        
        f_time = interp1d(et_time_interp, eeg_time_interp, fill_value='extrapolate')
        tvec_eeg = f_time(eyeTime)
    
        # Check what the times for each event are now and compare for later error-checking
        ttls_df['tvec_eeg'] = tvec_eeg[ttls_df['et_nearest_sample_index']]
        ttls_df['tvec_ttl_eeg_diff'] = np.abs(ttls_df['eeg_ttl_time'] - 
                                              ttls_df['tvec_eeg'])

        #############################################################
        # Aggregate data
        #############################################################
        et_data = {
            'right': eyeRight,
            'left': eyeLeft,
            'timestamps': tvec_eeg
        }
        
        
        # adjusting frame_times_sec so it starts at the movie onset
        event2use = 'movie_onset'  # or your relevant event
        event_time_align = ttls_df[ttls_df['events'] == event2use]['tvec_eeg'].values[0]
        frame_times_sec += event_time_align

    
    if task == 'fixation':
        
        # Number of eye-tracker samples
        nsamples = len(eyeTime)
        
        #############################################################
        # Align events and create time-vector for eye-tracker data
        #############################################################
        event2use = 'onset' # Event to use for alignment: use 'movie_onset', 'movie_offset' or 'start_calibration'
        df_event_row = np.where(ttls_df['events'] == event2use)[0][0]
        event_samplepoint_align = ttls_df.loc[[df_event_row],['et_nearest_sample_index']].values[0][0]
        event_time_align = ttls_df.loc[[df_event_row],['eeg_ttl_time']].values[0][0]
        
        # Interpolate the time axes to account for drift
        f_time = interp1d(ttls_df.et_ttl_time.values, ttls_df.eeg_ttl_time.values, fill_value='extrapolate')
        tvec_eeg = f_time(eyeTime)
        
        # Check what the times for each event are now and compare for later error-checking
        ttls_df['tvec_eeg'] = tvec_eeg[ttls_df['et_nearest_sample_index']]
        ttls_df['tvec_ttl_eeg_diff'] = np.abs(ttls_df['eeg_ttl_time'] - 
                                              ttls_df['tvec_eeg'])

        #############################################################
        # Aggregate data
        #############################################################
        et_data = {
            'right': eyeRight,
            'left': eyeLeft,
            'timestamps': tvec_eeg
        }
              
    if task == 'freeview':
        
        # Check what the times for each event are now and compare for later error-checking
        ttls_df['tvec_eeg'] = 0
        ttls_df['tvec_ttl_eeg_diff'] = 0
        ttls_df['et_nearest_sample_index_all'] = 0
        
        # Where to store all the data
        et_data = {
            'right': np.empty((0,13)),
            'left': np.empty((0,13)),
            'timestamps': np.empty((0))
        }
    
        # Event dictionary for later
        event_dict = {'onset': [], 'stim': [], 'offset': [], 'event': []}
        
        # Interpolate iEEG triggers if necessary
        eeg_ttls_interp = ttls_df['eeg_ttl_time'].values
        
        idx_nan = np.isnan(ttls_df.eeg_ttl_time).values
        
        f = interp1d(ttls_df['et_ttl_time'][np.invert(idx_nan)], 
                     ttls_df['eeg_ttl_time'][np.invert(idx_nan)],
                     fill_value='extrapolate')
        
        eeg_ttls_interp[idx_nan] = f(ttls_df['et_ttl_time'][idx_nan])
        
        #############################################################
        # Loop through each trial and aggregate data to a single vector
        #############################################################
        for trl in range(ntrials):
    
            # This trial's data
            r_data = eyeRight[trl]
            l_data = eyeLeft[trl]
            tvec_et = eyeTime[trl]
            trig_onset_et = ttls_df['et_ttl_time'][trl]
            trig_onset_eeg = eeg_ttls_interp[trl]
            event_label = ttls_df['events'][trl]
            stim_label = ttls_df['stim'][trl]
            
            # Add events that occurred this trial to dictionary
            event_dict['onset'].append(trig_onset_eeg)
            event_dict['offset'].append(trig_onset_eeg + stimulus_time)
            event_dict['event'].append(event_label)
            event_dict['stim'].append(stim_label)
            
            # Look for the eye-tracker sample point of image onset
            et_time_diff = np.abs(tvec_et - trig_onset_et)
            et_trig_onset_idx = et_time_diff.argmin()
            ttls_df.loc[trl,'et_nearest_sample_index'] = et_trig_onset_idx + et_data['timestamps'].shape[0]
            ttls_df.loc[trl,'et_nearest_timepoint'] = tvec_et[et_trig_onset_idx]
            ttls_df.loc[trl,'et_time_diff'] = et_time_diff.min()/10**6
            
            # Number of samples in this trial
            nsamples = len(tvec_et)
            
            # Interpolate the time axes to account for drift
            f_time = interp1d(ttls_df.et_ttl_time.values, ttls_df.eeg_ttl_time.values, fill_value='extrapolate')
            tvec_eeg = f_time(tvec_et)
              
            if np.any(tvec_eeg[0] < et_data['timestamps']):
                print(trl)
            
            # For later error checking
            ttls_df.loc[trl,'tvec_eeg'] = tvec_eeg[et_trig_onset_idx]
            ttls_df.loc[trl,'tvec_ttl_eeg_diff'] = np.abs(ttls_df.loc[trl,'tvec_eeg'] - trig_onset_eeg)
            
            #############################################################
            # Aggregate data
            #############################################################
            et_data['right'] = np.concatenate((et_data['right'],r_data))
            et_data['left'] = np.concatenate((et_data['left'],l_data))
            et_data['timestamps'] = np.concatenate((et_data['timestamps'],tvec_eeg))
         
        
        #############################################################    
        # Check timestamps are sorted
        #############################################################
        if not issorted(et_data['timestamps']):
            sys.exit('Something is wrong with the eye-tracker timestamps... they are not in ascending order')
                    
    # %% BUILD Perform error checks
    
    #############################################################
    # Check for any timeskips/gaps in the eye-tracker data
    #############################################################
    
    # Get rid of TTL code 255 (movie label, and fixation onset, offset. leave only movie onset, movie offset)
    task_event_indices = ttls_df['events'].isin(['movie_id', 
                                                 'fixation_start', 
                                                 'fixation_end',
                                                 'eyetracker_callback']) == False   
    ttls_df = ttls_df[task_event_indices]
    ttls_df = ttls_df.reset_index(drop=True)
    
    ttls_df['is_good'] = False
    ttls_df['notes'] = 'NA'
    
    # Check for gaps in data
    max_diff_time = (1/et_fs)*2 # Maximum allowed difference between two timepoints
    time_diffs = np.diff(tvec_eeg) # Time between timepoints
    gaps_idx = np.where( time_diffs >= max_diff_time )[0] # Where gaps in the data could be
    
    # Store this info in a dataframe
    gaps_df = pd.DataFrame({'t1': [], 't2': [], 'diff': [], 'nsamples': [],'t1_idx': []})
    for g in gaps_idx:
        gap = pd.DataFrame({
            't1': tvec_eeg[g],
            't2': tvec_eeg[g+1],
            'diff': tvec_eeg[g+1] - tvec_eeg[g],
            'nsamples': (tvec_eeg[g+1] - tvec_eeg[g])*et_fs,
            't1_idx': g},
            index=[0]
            )
        gaps_df = pd.concat([gaps_df, gap])
    
    # Create a string to give info about known gaps in the data
    gap_info_string = ''
    if not gaps_df.empty:
        gap_info_string = 'please note that there are gaps in the data that occur at these times (in seconds): '
        gaps_df['string'] = gaps_df['t1'].round(2).astype(str) + '-' + gaps_df['t2'].round(2).astype(str)
        gap_time_string = ', '.join(gaps_df['string'].tolist())
        gap_info_string += gap_time_string
        print('--->' + gap_info_string)
        
    #############################################################
    # Check each individual pulse
    #############################################################
    
    # Now go through each pulse to see if alignment is good
    for idx, row in ttls_df.iterrows():
        
        diff_is_good = row['tvec_ttl_eeg_diff'] <= max_diff_time
        notes2write = 'NA'
        
        # If time difference isn't good, first check if it's when a notable gap is
        if not diff_is_good and not gaps_df.empty:
            onset_tvec = row['tvec_eeg']
            isin_gap = np.logical_and(onset_tvec >= gaps_df['t1'], onset_tvec <= gaps_df['t2'])
            if isin_gap.any():
                diff_is_good = True
                notes2write = 'Occurs in a gap in eye-tracker time'
    
        # Check if this is the calibration offset
        if not diff_is_good and row['events'] == 'eye_tracker_off':
            if row['et_nearest_sample_index'] == (nsamples-1):
                diff_is_good = True
                notes2write = 'eye-tracker stopped recording before pulse sent'
                
        # Check if the iEEG pulse was missing
        if not diff_is_good and np.isnan(row['eeg_ttl_time']):
            diff_is_good = True
            notes2write = 'iEEG TTL was not recorded'
            
        if not diff_is_good:
            notes2write = 'offset in timing for unknown reason'
            
        # Add info to dataframe
        ttls_df.loc[idx,['is_good']] = diff_is_good
        ttls_df.loc[idx,['notes']] = notes2write
            
    # check all events are aligned
    events_timing_good = ttls_df['is_good'].all()
    if not events_timing_good:
        sys.exit('Something is wrong in the event timing, gotta debug')
    
    
    
        
    #%% Update TTL codes in nwb
    
    if task == 'freeview':      
        labels = ['image_onset']      
    elif task == 'movie':     
        labels = ['movie_onset', 'movie_offset', 'alignment']
    elif task == 'fixation':     
        labels = ['onset', 'offset']
    
    # Get ids for each code
    ttl_data = np.zeros(len(ttls_df)) * np.nan
    
    for l in range(len(labels)):
        ttl_data[ttls_df['events'] == labels[l]] = l
        
    ttl_data = ttl_data.astype(int)
    
    # Use iEEG TTLs and interpolate if necessary
    idx_nan = np.isnan(ttls_df.eeg_ttl_time).values
    
    f = interp1d(ttls_df['tvec_eeg'][np.invert(idx_nan)], 
                 ttls_df['eeg_ttl_time'][np.invert(idx_nan)],
                 fill_value='extrapolate')
    
    ttls_df.loc[idx_nan, 'eeg_ttl_time'] = f(ttls_df['tvec_eeg'][idx_nan])
    
    # Create new TTLs container
    new_container = TTLs(
        name=ttl_container.name,
        description=ttl_container.description,
        timestamps=ttls_df['eeg_ttl_time'].to_numpy(),
        data=ttl_data,
        labels=labels)
    
    nwb.acquisition.pop('TTL')
    nwb.add_acquisition(new_container)
    
    # Need to write and reload the nwb file
    new_fname = nwb_fname.replace('.nwb', '_new.nwb')
    
    with NWBHDF5IO(new_fname,mode='w') as export_io:
        export_io.export(src_io=io, nwbfile=nwb)
        
    # Rename and delete
    io.close()
    os.remove(nwb_fname)
    os.rename(new_fname,nwb_fname)
    
    # Reload again
    io = NWBHDF5IO(nwb_fname,mode='r+',load_namespaces=True)
    nwb = io.read()
    
    
    # %% BUILD Add eye-tracking data to the NWB file
    
    print('--->Adding eye-tracking data to file')
    
    # Create the processing module
    proc_mod = nwb.create_processing_module('eye_tracking','eye-tracking data collected using the Tobii TX300 eye-tracker')
    
    # Create container for eye-tracking data (not pupil diameter)
    et_container = EyeTracking(name='eyes')
    
    # Create container for pupil-tracking data
    pupil_container = PupilTracking(name='pupils')
    pupil_comments = 'eye-tracker uses a separate internal clock from other acquisitions'
    pupil_comments = pupil_comments + '; ' + gap_info_string
    
    # Description of reference frame for UCS (used across many containers)
    ucs_reference_frame_desc = """
    origin is the frontal surface of the eye-tracker device,
    the device is positioned below the monitor,
    x-axis is left/right (positive is right),
    y-axis is up/down (positive is up), z-axis is to and from the
    participant (positive is toward the participant)
    """.replace('\n',' ')
        
    #############################################################
    # Set Validity labels for "control" field
    #############################################################
    
    # Assign numbers and convert to uint8
    r_validity = et_data['right'][:,12]
    l_validity = et_data['left'][:,12]
    no_eyes = np.logical_and(r_validity == 4, l_validity == 4)
    r_validity[no_eyes] = 5
    l_validity[no_eyes] = 5
    r_validity = r_validity.astype('uint8')
    l_validity = l_validity.astype('uint8')
    
    # What the numbers represent for each eye
    r_validity_labels = [
    'found right eye',
    'found one eye, probably right eye',
    'found one eye but uncertain if left or right',
    'found one eye, probably left eye',
    'found one eye, most likely left eye',
    'no eyes found']
    
    l_validity_labels = [
    'found left eye',
    'found one eye, probably left eye',
    'found one eye but uncertain if left or right',
    'found one eye, probably right eye',
    'found one eye, most likely right eye',
    'no eyes found']
    
    
    ###################################################
    # Pupil
    ###################################################
    
    # Right pupil diameter TS
    pupil_container.create_timeseries(
        'r_eye',
        wrap_data(et_data['right'][:,11]),
        unit='millimeter',
        timestamps = et_data['timestamps'],
        description = 'diameter of the right pupil',
        comments = pupil_comments,
        control = r_validity,
        control_description = r_validity_labels
    )
    
    # Left pupil diameter TS
    pupil_container.create_timeseries(
        'l_eye',
        wrap_data(et_data['left'][:,11]),
        unit='millimeter',
        timestamps = et_data['timestamps'],
        description = 'diameter of the left pupil',
        comments = pupil_comments,
        control = l_validity,
        control_description = l_validity_labels
    )
    
    ###################################################
    # 2D Eye Coordinates ADCS
    ###################################################
    
    adcs_description = """
    xy position of the monitor the eye is fixated on in the
    "active display coordinate system". For more information, check out
    http://developer.tobiipro.com/commonconcepts/coordinatesystems.html
    """.replace('\n',' ')
    
    adcs_comments = pupil_comments
    
    adcs_reference_frame_desc = """
    (0,0) is the upper left corner of the monitor and (1,1) is the bottom right
    corner of the monitor. For more information, check out
    http://developer.tobiipro.com/commonconcepts/coordinatesystems.html
    """.replace('\n',' ')
    
    # Right eye 2D coordinates
    et_container.create_spatial_series(
        'r_eye_adcs',
        wrap_data(et_data['right'][:,6:8]),
        unit='millimeter',
        timestamps=et_data['timestamps'],
        description=adcs_description,
        comments=adcs_comments,
        control=r_validity,
        control_description=r_validity_labels,
        reference_frame = adcs_reference_frame_desc
    )
    
    # Left eye 2D coordinates
    et_container.create_spatial_series(
        'l_eye_adcs',
        wrap_data(et_data['left'][:,6:8]),
        unit='millimeter',
        timestamps=et_data['timestamps'],
        description=adcs_description,
        comments=adcs_comments,
        control=l_validity,
        control_description=l_validity_labels,
        reference_frame = adcs_reference_frame_desc
    )
    
    ###################################################
    # 3D Eye Coordinates UCS
    ###################################################
    
    ucs_eye_description = """
    xyz position of the eye in 3D space, part of the
    "user coordinate system" (UCS), use in conjunction with
    UCS gaze position for the "gaze vector".
    For more information, check out
    http://developer.tobiipro.com/commonconcepts/coordinatesystems.html
    """.replace('\n',' ')
    
    eye_coord_3d_comments = pupil_comments
    
    # Right eye 3D coordinates
    et_container.create_spatial_series(
        'r_eye_pos',
        wrap_data(et_data['right'][:,0:3]),
        unit='millimeter',
        timestamps=et_data['timestamps'],
        description=ucs_eye_description,
        comments=eye_coord_3d_comments,
        control=r_validity,
        control_description=r_validity_labels,
        reference_frame = ucs_reference_frame_desc
    )
    
    # Left eye 3D coordinates
    et_container.create_spatial_series(
        'l_eye_pos',
        wrap_data(et_data['left'][:,0:3]),
        unit='millimeter',
        timestamps=et_data['timestamps'],
        description=ucs_eye_description,
        comments=eye_coord_3d_comments,
        control=l_validity,
        control_description=l_validity_labels,
        reference_frame = ucs_reference_frame_desc
    )
    
    ###################################################
    # 3D Eye Gaze UCS
    ###################################################
    
    ucs_gaze_description = """
    xyz position on the calibration plane at which the eye is looking,
    this is part of what is called the "user coordinate system" (UCS),
    use in conjunction with eye position for the "gaze vector".
    For more information check out
    http://developer.tobiipro.com/commonconcepts/coordinatesystems.html
    """.replace('\n',' ')
    
    eye_gaze_comments = pupil_comments
    
    # Right eye gaze position
    et_container.create_spatial_series(
        'r_eye_gaze',
        wrap_data(et_data['right'][:,8:11]),
        unit='millimeter',
        timestamps=et_data['timestamps'],
        description=ucs_gaze_description,
        comments=eye_gaze_comments,
        control=r_validity,
        control_description=r_validity_labels,
        reference_frame = ucs_reference_frame_desc
    )
    
    # Left eye gaze position
    et_container.create_spatial_series(
        'l_eye_gaze',
        wrap_data(et_data['left'][:,8:11]),
        unit='millimeter',
        timestamps=et_data['timestamps'],
        description=ucs_gaze_description,
        comments=eye_gaze_comments,
        control=l_validity,
        control_description=l_validity_labels,
        reference_frame = ucs_reference_frame_desc
    )
    
    
    ###################################################
    # Track-Box Coordinate System
    ###################################################
    
    tbcs_description = """
    xyz position of the eye in the box the eye tracker can theoretically
    track the eye. This is called the "track box coordinate system".
    Units are normalized (0-1). For more information, check out
    http://developer.tobiipro.com/commonconcepts/coordinatesystems.html
    """.replace('\n',' ')
    
    tbcs_comments = pupil_comments
    
    tbcs_ref_frame = 'top-right corner of the box in which the eyes can be tracked'
    
    # Right eye TBCS coordinates
    et_container.create_spatial_series(
        'r_eye_tbcs',
        wrap_data(et_data['right'][:,3:6]),
        unit='normalized',
        timestamps=et_data['timestamps'],
        description=tbcs_description,
        comments=tbcs_comments,
        control=r_validity,
        control_description=r_validity_labels,
        reference_frame = tbcs_ref_frame
    )
    
    # Left eye TBCS coordinates
    et_container.create_spatial_series(
        'l_eye_tbcs',
        wrap_data(et_data['left'][:,3:6]),
        unit='normalized',
        timestamps=et_data['timestamps'],
        description=tbcs_description,
        comments=tbcs_comments,
        control=l_validity,
        control_description=l_validity_labels,
        reference_frame = tbcs_ref_frame
    )
    
    ###################################################
    # Add the Containers to Processing Module
    ###################################################
    
    proc_mod.add(et_container)
    proc_mod.add(pupil_container)
    
    ###################################################
    # Eye-Tracker Device
    ##################################################
    
    device_description.format(monitor_inches,monitor_res[0],monitor_res[1])
    nwb.create_device(
        'tobii_eye_tracker',
        description=device_description.format(monitor_inches,monitor_res[0],monitor_res[1]),
        manufacturer='Tobii'
    )
    
    # %% BUILD Add events to trial container
    # https://pynwb.readthedocs.io/en/latest/tutorials/domain/images.html
    
    # Events for FREEVIEW
    if task=='freeview':
 
        stim_files = ['stimuli/freeviewing/{:s}.png'.format(s) for s in event_dict['stim']]      
        timestamps = event_dict['onset']
        
        fv_comment = ('Each image is presented framed by a black border that takes up 10% of screen size; '
                      'the top left corner is at pixel [192,108] '
                      'and the bottom right and pixel [1728,972]; '
                      'the frame size presented is not the same as in the file, '
                      'but 1536x864 px on a 1920x1080 px screen with '
                      'physical dimensions of 509.2x286.4 mm')
       
        behavior_external_file = ImageSeries(
            comments=fv_comment,
            name="ExternalFiles",
            description="image files presented to the patient",
            unit="n.a.",
            external_file=stim_files,
            format="external",
            starting_frame=np.arange(len(stim_files)),
            timestamps=timestamps,
        )
        
        nwb.add_stimulus(behavior_external_file)
        
    # Events for MOVIE
    elif task=='movie':
        
        timestamps = et_data['timestamps'][frame_sample]
        external_file = ['stimuli/movies/{:s}'.format(mov_file)]
        movie_comment = ('movie is presented framed by a black border that takes up 10% of screen size; '
                         'the top left corner is at pixel [192,108] '
                         'and the bottom right and pixel [1728,972]; '
                         'the frame size presented is not the same as in the file, '
                         'but 1536x864 px on a 1920x1080 px screen with '
                         'physical dimensions of 509.2x286.4 mm')
        
        behavior_external_file = ImageSeries(
            comments=movie_comment,
            name="ExternalFiles",
            description="video presented to the patient; timestamps are the onset of individual frames in the movie",
            unit="n.a.",
            external_file=external_file,
            format="external",
            starting_frame=[0],
            timestamps=timestamps,
        )
        nwb.add_stimulus(behavior_external_file)

    # Events for MOVIE
    if task=='movie':
        nwb.add_trial_column(name='frame', description='frame number of the movie displayed')
        for ii in range(len(frame_times_sec)):
            nwb.add_trial(
                start_time = frame_times_sec[ii],
                stop_time = frame_times_sec[ii] + time_per_frame,
                frame = ii + 1
            )
                
                

    #no events for FIXATION
    elif task=='fixation':
        pass
     
    # %% BUILD Save and close NWB file
    print('--->Saving file')
    io.write(nwb)
    io.close()
       
# %% Run the script

if __name__ == '__main__':
    task = sys.argv[1]
    nwbFile = sys.argv[2]
    etFile = sys.argv[3]
    run(task,nwbFile,etFile)

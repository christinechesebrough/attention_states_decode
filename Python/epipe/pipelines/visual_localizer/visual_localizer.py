#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This is a template for HBML Visual Localizer analysis. Please do not
use the script found in the EPIPE directory to do this analysis. Instead,
please run `epipe.pipelines.visual_localizer.create_copy()` to create
a copy of the script that can then be altered at will.

This script performs the following functions:
    - Read in raw data from an NWB file
    - Extract TTLs or other event markers and align events to neural time
    - Preprocess
        * notch filter
        * downsample
    - Bad channel rejection
    - Rereference (average and bipolar reference)
        * All operations below here are done on both average and
            bipolar reference data
    - Epoch and epoch rejection
    - Filter HFA and calculate t-stat
    - Time-frequency analysis using morlet wavelets
    - Generate html report file

Before starting, please examine the variables in the first section
that set important paths such as where data will be saved and where
to find the freesurfer directory.

This script can be run either step by step or by simply hitting "run". However,
troubleshooting may need to be done, particularly at the beginning of the script,
where data is being loaded and events are being extracted. If the data being used
was acquired by unusual means then this will most likely be the case.

Happy analysing!

Noah Markowitz
Human Brain Mapping Laboratory
North Shore University Hospital
January 2022
"""

# Try this link for more info on constructing a pipeline 
# https://mne.tools/stable/auto_tutorials/intro/10_overview.html#sphx-glr-auto-tutorials-intro-10-overview-py

# Import directories

# these are suggested matplotlib specifications
import matplotlib
matplotlib.interactive(False)
matplotlib.use('qtagg')

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import os
import os.path as op
import pandas as pd
import re
from collections import OrderedDict
from datetime import datetime
from pynwb import NWBHDF5IO
from tqdm import tqdm
from sklearn.preprocessing import minmax_scale
import pathlib
import nibabel as nib
from mne.time_frequency import tfr_morlet, write_tfrs
from mne import Epochs
import mne # Any other needed MNE modules and function can be loaded from here
from epipe import (load_nwb_settings, inspectNwb, nwb2mne, create_toi_slice,
                   reref_avg, reref_bipolar, filter_hfa_continuous, filter_hfa_epochs,
                   ana2dig, create_mne_report_table, plot_ortho, plot_brain_surf,
                   plot_stem_annotation)

# Set some stuff for mne
mne.viz.set_browser_backend("pyqtgraph")
mne.viz.set_3d_backend('pyvistaqt')
mne.set_log_level('ERROR')

# Pipeline info
pipeline_name = 'mne-python-visual_localizer'
pipeline_version = '2023.12.08'

# Load NWB settings as written in the json file
nwbSettings = load_nwb_settings()
bad_chan_params = nwbSettings['ecog_file']['fields']

#region Settings to edit



###########################################################################
# SETTINGS TO EDIT
###########################################################################

# You can change variables and filenames at will. These are just set
# for convenience and to keep all output together

# The number of cores that can be used at once for some operations
# Increasing this will speed up some processes
# # Just make sure not all cores on computer are being used. Leave at least 1 free
n_jobs = 4

########################### FILENAMES ###########################

# Set "code_file" to be the location of this file. A copy of this file will be added to
# the html file reports that are generated
code_file = "/media/hbml/HDD2/Lisa/VisLoc/visual_localizer_template.py"

# a top-level directory to hold data and save output
rootDir = "/media/hbml/HDD2/Lisa/VisLoc/test"

# name of the nwb file containing raw data or the one to create from the raw data file (listed above)
# Should be named according to BIDS convention
nwbFname = rootDir + os.sep + 'sub-NS194_ses-implant01_task-visloc_run-01_acq-spanish_ieeg.nwb'
# the log file containing events
logFname = os.path.join(rootDir, 'NS194_B13_VIsLoc', 'NS194_B13_VIsLoc_Run1_spanish.csv')       # first is folder with beh data, second is the filename of the log for the block you're looking at

# Freesurfer directory and freesurfer id of the subject
freesurfer_subjects_dir = "/home/hbml/freesurfer/subjects"
fs_subid = 'NS194'
subFsDir = freesurfer_subjects_dir + os.sep + fs_subid
brainmask_fname = os.path.join(subFsDir, 'mri', 'brainmask.mgz')
brainmask = nib.load(brainmask_fname)

# Info that can be extracted from the naming of the NWB file
# shouldn't need changing
parts_list = os.path.basename(nwbFname).split('_')[:-1]
fparts = {v.split('-')[0]: v.split('-')[1] for v in parts_list}
subid = fparts['sub']
sesid = fparts['ses']
full_task_name = fparts['task']
if 'acq' in fparts.keys():
    full_task_name = (full_task_name + '_' + fparts['acq'])
pipelineDir = os.path.join(rootDir,'derivatives', pipeline_name)
resultsDir = os.path.join(pipelineDir, 'sub-' + subid, 'ses-' + sesid, full_task_name)
baseFname = resultsDir + os.sep + '_'.join(parts_list)
baseReportFname = pipelineDir + os.sep + '_'.join(parts_list)
hfa_fname = os.path.join(resultsDir, 'sub-' + subid + '_ses-' + sesid + '_' + full_task_name + '_hfa.fif')


# How to name output files. These can generally be left alone
filenames = {
    'preproc': baseFname + '_' + 'desc-preproc_ieeg.fif.gz',
    'events': baseFname + '_' + 'eve.txt',
    'reref': baseFname + '_' + 'ref-{ref}_ieeg.fif.gz',
    'hfa': baseFname + '_' + 'ref-{ref}_desc-hfa_ieeg.fif.gz',
    'epochs': baseFname + '_' + 'ref-{ref}_epo.fif.gz',
    #'hfa': baseFname + '_' + 'ref-{ref}_desc-hfa_ave.fif.gz',
    'tfr': baseFname + '_' + 'ref-{ref}_desc-morlet_tfr.h5',
    'report': baseReportFname + '_' + 'ref-{ref}_report.html'
}

# Add the path to this file so that a copy of this code can be added to the report generated
code_file_path = pathlib.Path(code_file)

# Make directory for output
if not os.path.isdir(resultsDir):
    os.makedirs(resultsDir)

########################### PREPROCESSING SETTINGS ###########################

# What to downsample data to in preprocessing
resample_fs = 500

# If trial rejection was done on a previous reference type, then
# the bad epochs can automatically be assigned as such in
# other references types without having to perform trial rejection again
# This is optional. One bonus of doing trial rejection again is that
# It allows you to examine raw traces of your rereferenced data
trial_rejection = {
    'reuse': False, # Set this to true if you only want to do trial rejection once
    'done': False,
    'idx': []
}

########################### REREFERNCING SETTINGS ###########################

# Types of references to use in ieeg_analyses
# Must be a list containing at least one of the options: "avg", "bip"
# bipolar referencing throws an error as of 12/8/2023
ref_types = ['avg']

########################### EPOCHING SETTINGS ###########################

# Duration of epochs. This can be increased for the time-frequency analysis
epoch_window = [-2, 2] # this is ±2

########################### HFA SETTINGS ###########################

# Interval for baseline correction for time-frequency analysis (old)
baseline_correction_time = (-0.35, -0.05)

# For the independent t-test, compare the average of these two windows
hfa_baseline_time = (-0.35, -0.05)
hfa_active_time = (0.1,0.3)

########################### TIME-FREQUENCY SETTINGS ###########################

# Frequencies to measure and number of cycles
# Number of cycles can be the same for all frequencies or can vary by frequency
freqs = np.logspace(*np.log10([4, 200]), num=70)  # Frequencies of interest
n_cycles = 5  # can also do freqs/2 for different number of cycle per frequency

# Programmatic decimation. Select specific timepoints to analyze as well as the start/stop points
# Below should work the same as the "toi" cfg in ft_freqanalysis
# Matlab equivalent of below is: [tstart:tstep:tend]
tfr_tstart = -2
tfr_tend = 2
tfr_tstep = 0.01

########################### REPORT SETTINGS ###########################

# Time being displayed in report figures
report_figures_time_window = (-0.05, 0.7)

# For TFR plots, the minimum and maximum frequencies to display
report_fmin = 4
report_fmax = 200

# What the type of scale for the y-axis should be used in the tfr plots in the report
report_scale_type = 'log'

#endregion

############################################################################
# END OF SETTINGS
# YOU TYPICALLY DONT HAVE TO CHANGE ANYTHING BEYOND THIS POINT
############################################################################

#region Load NWB file and put in MNE format

print('--->Reading file: %s' % nwbFname)

# NWB read
io = NWBHDF5IO(nwbFname, mode='r+', load_namespaces=True)
nwb = io.read()

# Get info on data in NWB file
nwbInfo = inspectNwb(nwb)
tsInfo = nwbInfo['timeseries']
elecTable = nwbInfo['elecs']

# Get ieeg data
if 'ieeg' in tsInfo['name'].to_list():
    ecogContainer = nwb.acquisition.get('ieeg')
    fs = ecogContainer.rate
    ecog = nwb2mne(ecogContainer,preload=False,create_montage=True)

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
    _, ttls = ana2dig(ana_ttls, fs=ttl_rate, min_diff=0.4, return_time=True)



# Add custom info for the report. This will be converted to a table and added to the report as meta-data
report_info = OrderedDict(
    Subject=fs_subid,
    Age=nwbInfo['subject']['age'],
    Sex=nwbInfo['subject']['sex'],
    Amplifier=nwbInfo['devices']['name'][0],
    Experiment='Visual Localizer',
    FullTaskName=full_task_name,
    Session=nwb.session_id,
    Session_Description=nwb.session_description,
    Experiment_Description=nwb.experiment_description,
    Experiment_Notes=nwb.notes,
    Lab=nwb.lab,
    Institution=nwb.institution,
    Original_File=os.path.basename(nwbFname),
    Pipeline=pipeline_name,
    PipelineVersion=pipeline_version,
    MNE_Version=mne.__version__,
    Analysis_Date=datetime.now().strftime('%Y-%m-%d'))

#endregion

#region Load log file and add to MNE format

print('--->Getting events')

# Import and filter for only images
logDf = pd.read_csv(logFname, comment='#')

# Basics
logDf['EventType'] = logDf['EventType'].str.lower()
logDf = logDf.rename(columns={'Time': 'compTime'})
logDf['neuralTime'] = 0
logDf['event'] = 0
logDf['keep_evt'] = True

# remove rows that contain a certain code. Only do so if there is not a column named 'TTL'
if not 'TTL' in logDf.keys():
    codes2Remove = ['fixation']
    rows2RemoveIdx = logDf['Code'].isin(codes2Remove)
    extraRows = logDf[rows2RemoveIdx]
    logDf = logDf[~rows2RemoveIdx]

# if there is a column 'TTL' in the logfile, keep only lines where ttl is 1
if 'TTL' in logDf.keys():
    logDf = logDf[logDf['TTL']==1]
    logDf.pop('TTL')

# if the codes in logfile start with 'VS13_', take only the second section after the first underscore
if logDf['Code'].iloc[0][0:4] == 'VS13':
    logDf['Code'][logDf['Code'].str.split('_').str.len()==3] = logDf['Code'].str.split('_').str[1]

# if there are codes of the pattern textStim_color_YELLOW: make them be named 'text'
if sum(logDf['Code'].str.contains('textStim_')) > 0:
    logDf['Code'][logDf['Code'].str.contains('textStim_')] = 'text'

# if there are numbers in the codes, remove them
if sum(logDf['Code'].str.contains('\d')) > 0:
    logDf['Code'] = logDf['Code'].str.extract('([a-z]+)')

# if the event type is marked as "pictureonset", make it "picture"
if "pictureonset" in logDf['EventType'].values:
    logDf['EventType'][logDf['EventType'] == 'pictureonset'] = 'picture'

# Definition for events for epoching
# Numeric values for each type of event
if 'face' in logDf['Code'].values:
    eventsDictionary = {'face': 1, 'house': 2, 'body': 3, 'object': 4, 'text': 5, 'pattern': 6}
elif 'people' in logDf['Code'].values:
    eventsDictionary = {'people': 1, 'places': 2, 'animals': 3, 'tools': 4, 'words': 5, 'patterns': 6}
else:
    raise ValueError('Something is wrong with how the logfile is read. The "Code" column is not containing what this script expects. Adapt the script to work with your version of a logfile!')

# check that the amount of TTLs in the logfile is the same as in the neural data
if ttls.shape[0] != logDf.shape[0]:
    if ttls.shape[0] - logDf.shape[0] == 23:
        raise Warning('it seems like the neural recording has both training and run 1. \n Disregarding the first 23 ttl pulses for this analysis. \n If you think this is not actually the case, please check why there are 23 extra triggers.')
        ttls = ttls[23:]
    else:
        raise ValueError('the number of ttl pulses in neural data and logfile do not lign up')

# Align TTLs to to neural time
ttl_idx = 0
last_stim_comp_time = 0
last_stim_neural_time = 0
exp_started = False
for idx, row in logDf.iterrows():
    evt_code = row['Code']
    evt_type = row["EventType"]
    if evt_code == 'Response' and not exp_started:
        logDf.loc[idx,"keep_evt"] = False
        continue
    elif evt_type == "picture":
        logDf.loc[idx,"neuralTime"] = ttls[ttl_idx]
        ttl_idx += 1
        for ev in eventsDictionary.keys():
            if ev in evt_code:
                logDf.loc[idx, 'event'] = eventsDictionary[ev]

        last_stim_comp_time = logDf.loc[idx, "compTime"]
        last_stim_neural_time = logDf.loc[idx, "neuralTime"]

        if not exp_started:
            exp_started = True

    else:
        time_diff = logDf.loc[idx, "compTime"] - last_stim_comp_time
        logDf.loc[idx, "neuralTime"] = last_stim_neural_time + time_diff


logDf = logDf[logDf["keep_evt"]]

# Compare
time_diff_comp = np.diff(logDf["compTime"])
time_diff_neural = np.diff(logDf["neuralTime"])
if not np.all(np.abs(time_diff_comp-time_diff_neural) < 0.001):
    raise ValueError("Some events may not be correctly aligned")

# Clean a little
logDf = logDf.drop(labels=["compTime", "keep_evt"], axis=1)
if logDf['Code'].to_list()[-1].lower() == "fixation":
    logDf = logDf[:-1]

# Format for MNE
mneDf = logDf.copy()
mneDf = mneDf[mneDf["Code"] != "Response"]
mneDf['sample'] = np.round(mneDf['neuralTime']*resample_fs).astype(int)
mneDf["dummy"] = 0
mneDf = mneDf.loc[:,['sample','dummy','event']]
eventsArray = mneDf.to_numpy()
eventsFname = filenames['events']
mne.write_events(eventsFname, eventsArray, overwrite=True)

# Format for NWB table
nwbDf = logDf.copy()
nwbDf = nwbDf.drop(labels=["Trial","event"], axis=1)
nwbDf.columns = [c.lower() for c in list(nwbDf.columns)]
nwbDf = nwbDf.rename(columns={'neuraltime': 'start_time',"eventtype": "event_type"})
nwbDf["stop_time"] = 0
for idx, row in nwbDf.iterrows():
    if row["event_type"] == "response":
        nwbDf.loc[idx,"stop_time"] = nwbDf.loc[idx,"start_time"]
    else:
        nwbDf.loc[idx, "stop_time"] = nwbDf.loc[idx, "start_time"] + 0.24


# Write to NWB file and close
nwb.add_trial_column(name="event_type", description="type of event")
nwb.add_trial_column(name="code", description="event code, type of stimulus is denoted by the string before number of underscore")
for idx, row in nwbDf.iterrows():
    nwb.add_trial(**row.to_dict())

io.write(nwb)
io.close()

#endregion


###########################################################################
# BEGIN ANALYSIS
###########################################################################


#region Notch filter, down sample, add/remove bad channels by inspecting raw trace, save

print('--->Applying notch filters and downsampling to %2.fHz' % resample_fs)

# Copy the `ecog` variable and then resample and apply notch filter
notch_freqs = (60, 120, 180)
ecogPreproc = ecog.resample(resample_fs).notch_filter(notch_freqs, notch_widths=2)
del ecog

# Display the raw traces and mark bad channels
nbadOrig = ecogPreproc.info['bads']
fig = ecogPreproc.plot(show=True, block=True, remove_dc=True, duration=15.0, n_channels=16)

# import h5py
# nwb_h5 = h5py.File(nwbFname, mode="r+")
# nwb_h5['/general/extracellular_ephys/electrodes/soz'][()]

# elecs = nwb.electrodes
# nwb.set_electrode_table()
# elecs.soz.data[()]
#
#



# Save the current state of the data in the MNE format
ecogPreproc.save(filenames['preproc'],fmt='single',overwrite=True)

#endregion

# This loop runs all other steps on all types of references specified to use
# ref_types: "avg", "bip"
for ref in ref_types:

    print('#'*50)
    print('Beginning processing for data using %s reference' % ref)
    print('#' * 50)

    # What the preprocessed filename for this reference type should be
    preprocRerefFname = filenames['reref'].format(ref=ref)

    # Check if the preprocessed file already exists so you don't have to redo rereferencing functions
    if os.path.isfile(preprocRerefFname):
        ecogReref = mne.io.read_raw_fif(preprocRerefFname, preload=True)
        del ecogPreproc
    else:
        if 'ecogPreproc' not in locals():
            ecogPreproc = mne.io.read_raw_fif(filenames['preproc'], preload=True)

        if ref == 'avg':
            ecogReref = reref_avg(ecogPreproc)

        elif ref == 'bip':
            ecogReref = reref_bipolar(ecogPreproc)

        # Save the referenced data in MNE format
        ecogReref.save(preprocRerefFname,fmt='single',overwrite=True)
        del ecogPreproc

    # Apply HFA filter
    ecogHfa = filter_hfa_continuous(ecogReref.copy(), hfa_fname, n_jobs=3)

    #region Epoch data

    print('--->Epoching')

    # Name of the MNE epoch file
    epochFname = filenames['epochs'].format(ref=ref)

    # Make meta-data that will be used later
    metadata, _, _ = mne.epochs.make_metadata(
        events=eventsArray, # Numpy array of events. Where each number corresponds to an event
        event_id=eventsDictionary, # Dictionary that relates number to string of events
        tmin=epoch_window[0],
        tmax=epoch_window[1],
        sfreq=ecogReref.info['sfreq'])

    # Create the epochs objects
    epochs = Epochs(
        ecogReref,
        eventsArray,
        tmin=epoch_window[0],
        tmax=epoch_window[1],
        event_id=eventsDictionary,
        reject_by_annotation=False,
        metadata=metadata,
        reject=None,
        flat=None,
        detrend=None)

    # Create epochs hfa object to tag along
    epochsHfa = Epochs(
        ecogHfa,
        eventsArray,
        tmin=epoch_window[0],
        tmax=epoch_window[1],
        event_id=eventsDictionary,
        reject_by_annotation=False,
        metadata=metadata,
        reject=None,
        flat=None,
        detrend=None)

    # Check epochs. If epoch rejection was already done on another reference type then reject those same epochs
    if trial_rejection['reuse'] and trial_rejection['done']:
        epochs.drop(trial_rejection['idx'])
        epochsHfa.drop(trial_rejection['idx'])
    else:

        # Databrowser for rejecting epochs
        fig = epochs.plot(
            n_epochs=5,
            show=True,
            block=True,
            n_channels=20)

        drops = epochs.drop_log
        trial_rejection['idx'] = [ii for ii in range(len(drops)) if len(drops[ii]) > 0]
        epochsHfa.drop(trial_rejection['idx'])
        trial_rejection['done'] = True

        # Save the epochs object
        epochs.save(epochFname, overwrite=True)

    #endregion

    #region TIME-FREQUENCY ANALYSIS (USING WAVELETS)

    print('--->Performing wavelet analysis')

    wltFname = filenames['tfr'].format(ref=ref,tfr='morlet')

    # Create a `slice` object. Allows more complex specification of wavelet timepoints to analyze
    # Ex: From -0.2 to 0.4 seconds analyze every 0.01 seconds (need it to be a slice object)
    toi = create_toi_slice(epochs.times, tfr_tstart, tfr_tend, tfr_tstep)

    # List to hold TFR objects
    morlet = []

    # Get a list of the different event types
    epoch_types = list(epochs.event_id.keys())

    # Perform morlet wavelet on each type of stimuli
    for stimii in tqdm(range(len(epoch_types)), unit='epo', desc='Performing tfr morlet', ncols=100):
        stim = epoch_types[stimii]
        pow, itc = tfr_morlet(
            epochs[stim],
            freqs=freqs,
            n_cycles=n_cycles,
            use_fft=True,
            average=True,
            return_itc=True,
            decim=toi,
            n_jobs=n_jobs, # How many processes can be done at once. Speeds up process
            picks=np.arange(0, len(epochs.ch_names), 1), # Which channels to apply this to
            output='power',  # Options: 'complex', 'power', `complex` produces LARGE amounts of data
            verbose='ERROR')

        # Add a small comment to the object describing what kind of data it is
        pow.comment = stim + '_pow'
        itc.comment = stim + '_itc'

        # Append both objects to the list
        morlet += [pow,itc]

    # write out tfr objects
    write_tfrs(wltFname, morlet, overwrite=True)
    tfr_idx = [t.comment for t in morlet]

    #endregion

    #region HFA over epoched data: Now doing this earlier in the script over continuous data

    # print('--->Filtering for HFA')
    #
    # # Filter for HFA
    # hfa = filter_hfa_epochs(epochs, n_jobs=n_jobs, baseline_time=baseline_correction_time, resample_fs=100)
    # hfa_idx = [t.comment for t in hfa]
    #
    # # Save the HFA data
    # hfaFname = filenames['hfa']
    # hfaFname = hfaFname.format(ref=ref)
    # mne.write_evokeds(hfaFname, hfa, overwrite=True)

    #endregion

    #region Calculate t-statistic for channels using HFA
    evokedHfa = epochsHfa.average(by_event_type=True)
    hfa_tvals = pd.DataFrame(index=epochsHfa.ch_names)
    hfa_norm = pd.DataFrame(index=epochsHfa.ch_names)
    for evokedStim in evokedHfa:

        # Get data
        stim_name = evokedStim.comment
        baseline_data = evokedStim.get_data(tmin=hfa_baseline_time[0], tmax=hfa_baseline_time[1])
        active_data = evokedStim.get_data(tmin=hfa_active_time[0], tmax=hfa_active_time[1])

        # Calculate hfa via normalization
        # hfa_power = active_data.copy().mean(axis=1)
        # hfa_power -= hfa_power.min()
        # hfa_power /= hfa_power.max()

        # Go through each channel and calculate t-statistic
        stim_tvals = []
        for chnii in range(baseline_data.shape[0]):
            t = mne.stats.ttest_ind_no_p(a=baseline_data[chnii,:], b=active_data[chnii, :])
            stim_tvals.append( abs(t) )

        # Add new column to DataFrame
        hfa_tvals[stim_name] = stim_tvals

    #endregion

    #region Start Report

    # Set Report file for output
    reportFile = filenames['report'].format(ref=ref, tfr='morlet')
    rep = mne.Report(title='Visloc ' + ref + ' reference')

    # Create a little table for the meta-data specified at the start of the script
    create_mne_report_table(report_info,report=rep,name='Metadata',tags=['Info'])

    # Add a small plot of events
    rep.add_events(eventsArray,'Events',event_id=eventsDictionary, sfreq=epochs.info['sfreq'],tags=['Info'])

    #endregion

    #region Generate figures for Wavelet and ERP
    print('--->Generating figures for html report')

    #chn_indices = np.arange(20,100,1)
    chn_indices = np.arange(0, len(epochs.ch_names), 1)
    montage = epochs.get_montage()
    elec_coords = montage.get_positions()['ch_pos']
    separate_fig_types = ['pow_bl', 'itc'] #['pow','pow_bl','itc']
    fig_sup_titles = {
        'pow_bl': 'Power: Baseline correction (seconds) %s' % str(baseline_correction_time),
        'pow': 'Power: No baseline correction',
        'itc': 'Inter-Trial Coherence'
    }

    # NOTE: The baseline_method "logratio" in MNE is the same as baselinetype="db" in fieldtrip's "ft_freqbaseline" function7y

    # Produce images for each electrode. All figures for one electrode will be combined so as
    # to be able to scroll between them with a slider
    # The "tqdm" function is just to show a progress bar
    for chnii in tqdm(chn_indices,unit='chn',desc='Generating Report Figures',ncols=100):
        chn = epochs.ch_names[chnii]
        figs = [] # List of figures relating to this electrode
        captions = []
        for fig_type in separate_fig_types:

            if fig_type == 'pow_bl':
                datatype='pow'
                baseline_method = 'logratio'
                use_db = False
                this_baseline_correction_time = baseline_correction_time
            elif fig_type == 'pow':
                datatype = 'pow'
                baseline_method = 'logratio'
                use_db = False
                this_baseline_correction_time = None
            else:
                datatype = 'itc'
                baseline_method = 'logratio'
                use_db = False
                this_baseline_correction_time = None

            matplotlib.interactive(False)
            thisFigTitle = chn + ' ' + fig_sup_titles[fig_type]
            thisFig, thisAxes = plt.subplots(2,3,figsize=(12,8))
            plt.show(block=False)
            plt.ioff()
            captions.append(thisFigTitle)


            geom = np.arange(6).reshape(2,3)
            leftFigs = geom[:,0]
            rightFigs = geom[:,-1]
            bottomFigs = geom[-1,:]
            topFigs = geom[0,:]

            thisAxes = thisAxes.flatten()
            stimii = -1

            # Go through each type of stimulus
            for stim in epochs.event_id.keys():

                # The TFR data for this particular stimulus
                data = morlet[tfr_idx.index(stim + '_' + datatype)]

                stimii+=1
                thisAx = thisAxes[stimii]

                # Get erp
                erpData = epochs[stim].average(chn).get_data(tmin=report_figures_time_window[0], tmax=report_figures_time_window[1]).T
                erp_tvec = np.linspace(report_figures_time_window[0], report_figures_time_window[1], erpData.size)

                # Get hfa data
                hfaData = epochsHfa[stim].average(picks=chn).get_data(chn, tmin=report_figures_time_window[0], tmax=report_figures_time_window[1]).T
                #hfaData = hfa[hfa_idx.index(stim)].get_data(chn, tmin=report_figures_time_window[0], tmax=report_figures_time_window[1]).T
                tvec_hfa = np.linspace(report_figures_time_window[0], report_figures_time_window[1], hfaData.size)

                # Power or ITC for this channel
                data.plot(
                    picks=chn,
                    mode='logratio', #baseline_method, # mean, ratio, logratio, percent, zscore, zlogratio
                    baseline=this_baseline_correction_time,
                    dB=use_db,
                    fmin=report_fmin,
                    fmax=report_fmax,
                    tmin=report_figures_time_window[0],
                    tmax=report_figures_time_window[1],
                    yscale=report_scale_type,
                    show=False,
                    axes=thisAx)
                plt.show(block=False)
                plt.ioff()

                # Add vertical dashed lines to denote stimulus onset and offset
                thisAx.axvline(0,color='green',linestyle='--')
                thisAx.axvline(0.24,color='black',linestyle='--')
                thisAx.set_title(stim)

                # Rescale the ERP and HFA data so that they can both fit in the image with ERP on top and HFA on bottom
                # Boundaries of the ERP and HFA waveforms
                erp_min, erp_max, hfa_min, hfa_max = np.percentile(thisAx.get_yticks(), [10, 30, 70, 90])

                # Overlay ERP on spectrogram
                erpData_rescale = minmax_scale(erpData, feature_range=(erp_min, erp_max) )
                thisAx.plot(erp_tvec, erpData_rescale, color='purple')

                # Overlay HFA on spectrogram
                hfaData_rescale = minmax_scale(hfaData, feature_range=(hfa_min, hfa_max))
                thisAx.plot(tvec_hfa, hfaData_rescale, color='gray')

                # If this figure is on the top row, don't include x-axis
                if stimii in topFigs:
                    thisAx.set_xticks([])
                    thisAx.set_xlabel('')

                # If this figure is not on the far left, don't display y-axis
                if stimii not in leftFigs:
                    thisAx.set_yticks([])
                    thisAx.set_ylabel('')

            #thisFig.tight_layout()

            # Add in the legend that tells when stimulus occurred and what ERP and HFA waveforms are
            legend_elements = [Line2D([0], [0], color='green', lw=1, linestyle='--', label='Onset'),
                               Line2D([0], [0], linestyle='--', color='black', label='Offset'),
                               Line2D([0], [0], color='purple', label='ERP'),
                               Line2D([0], [0], color='gray', label='HFA')]
            thisFig.legend(handles=legend_elements, loc='right')
            thisFig.suptitle(thisFigTitle, y=0.93)

            # Append and close
            figs.append(thisFig)
            plt.close(thisFig)

        #region Also add a figure showing channel location (IGNORE)
        if not np.isnan(elec_coords[chn][0]) and not np.all(elec_coords[chn] == elec_coords[chn][0]):
            plt.ioff()
            ortho_fig = plot_ortho(
                volume=brainmask,
                subject=fs_subid,
                subjects_dir=freesurfer_subjects_dir,
                coord=elec_coords[chn],
                elec_name=chn,
            )
            figs.append(ortho_fig)
            captions.append(chn + ' ortho view')
            plt.close("all")
        #endregion

        # Get tags for this electrode to help later with
        grp_name = re.sub(r'\d+$', '', chn)
        plt_tags = [grp_name,'TFR']
        if chn in epochs.info['bads']:
            plt_tags.append('bads')

        # Add the list of figures to the html file. When added as a list you'll be able to scroll through images with a slider
        rep.add_figure(figs, title=chn, tags=plt_tags, caption=captions, section=grp_name)

    #endregion

    #region Generate stem plots for t-values
    hfa_stem_plots = []
    hfa_tval_caption_text = "Absolute value of t-statistic for %s stimuli calculated using an independent samples t-test"\ 
    "comparing mean HFA at baseline (%0.2fs to %0.2fs) and mean HFA after stimulus presented (%0.2fs to %0.2fs)"
    hfa_tval_caption_text % ("face", hfa_baseline_time[0], hfa_baseline_time[1], hfa_active_time[0], hfa_active_time[1])
    captions = []
    for stim in epochs.event_id.keys():
        tvals = hfa_tvals[stim].to_numpy()
        tval_threshold = tvals.mean() + 3*tvals.std()
        hfa_tval_plot = plot_stem_annotation(tvals, list(hfa_tvals.index), threshold=tval_threshold, title=stim + " HFA t-value per channel")
        hfa_stem_plots.append(hfa_tval_plot)
        this_caption = hfa_tval_caption_text % (stim, hfa_baseline_time[0], hfa_baseline_time[1], hfa_active_time[0], hfa_active_time[1])
        captions.append(this_caption)
        plt.close("all")
    rep.add_figure(hfa_stem_plots, title="HFA t-values", tags=["HFA"], caption=captions)
    #endregion

    #region Add 3D figure of brain to report
    # currently doesn't produce anything meaningful because it's not finished yet, so I commented out.

    # brain_pial_fig = plot_brain_surf(
        # subject=fs_subid,
        # subjects_dir=freesurfer_subjects_dir,
        # views="omni",
        # surf="pial",
        # annotations=True,
        # elec_size=0.4
    # )
    # brain_pial_dk_fig = plot_brain_surf(
        # subject=fs_subid,
        # subjects_dir=freesurfer_subjects_dir,
        # views="omni",
        # parc="dk",
        # surf="pial",
        # annotations=True,
        # elec_size=0.4
    # )
    # brain_inf_fig = plot_brain_surf(
        # subject=fs_subid,
        # subjects_dir=freesurfer_subjects_dir,
        # views="omni",
        # surf="inflated",
        # snap_to_surf=True,
        # annotations=True,
        # max_dist=4,
        # elec_size=0.4
    # )
    # brain_inf_dk_fig = plot_brain_surf(
        # subject=fs_subid,
        # subjects_dir=freesurfer_subjects_dir,
        # views="omni",
        # parc="dk",
        # surf="inflated",
        # annotations=True,
        # snap_to_surf=True,
        # max_dist=4,
        # elec_size=0.4
    # )
    # surf_figs = [brain_pial_fig, brain_pial_dk_fig, brain_inf_fig, brain_inf_dk_fig]
    # rep.add_figure(surf_figs, title="Brain plots", tags=["surf_plots"], caption=["pial", "pial+dk", "inflated", "inflated+dk"])
    # #endregion

    #region Add code to report
    # Adds a copy of the code file (this file) used to analyze the data to the html file
    rep.add_code(code_file_path, 'Copy of analysis', language='python', tags=['code'])
    #endregionv

    # Save report
    print('--->Saving html report: %s' % reportFile)
    rep.save(reportFile, overwrite=True)


print('--->Visual Localizer Analysis Complete!')

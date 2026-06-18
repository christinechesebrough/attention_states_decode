#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 25 13:50:20 2025

@author: Maximilian Nentwich

Add speech features to NWB files
"""
from pathlib import Path
import numpy as np
from scipy.io import wavfile
from scipy import signal
from pynwb import NWBHDF5IO
from pynwb.behavior import IntervalSeries
#import textgrids

# Function to add textgrid tiers to IntervalSeries
def add_text_to_module(text_grid, tier_name, t_shift):
    
    tier_array = text_grid.interval_tier_to_array(tier_name)
    
    tier_onset = np.array([a['begin'] for a in tier_array])    
    tier_onset = tier_onset + t_shift 
    
    tier_offset = np.array([a['end'] for a in tier_array])    
    tier_offset = tier_offset + t_shift 
    
    tier_data = np.stack((np.ones(len(tier_onset)), -1*np.ones(len(tier_offset)))).T.flatten()
    time_vec = np.stack((tier_onset, tier_offset)).T.flatten()
    
    if tier_name == 'phones':
        tier_name = 'phonemes'
        
    container = IntervalSeries(tier_name, 
                                    data = tier_data,
                                    timestamps = time_vec,
                                    description = 'Time of {:s} in the presented audio data; 1:onset, -1:offset'.format(tier_name))
        
    return container
    

def run(nwb_fname):
    
    # Load recorded audio from NWB file
    io = NWBHDF5IO(nwb_fname, mode='r+', load_namespaces=True)
    nwb = io.read()
    
    if 'speech_annotation' in nwb.processing:
        io.close()
        return
    
    audio_data = nwb.acquisition['audio'].data[:][:,0]
    fs_audio = nwb.acquisition['audio'].rate
    
    # Create a new module
    annot_module = nwb.create_processing_module('speech_annotation', 'Annotated word and phoneme onset and offset')
    
    # Load the original audio from the file 
    current_dir = Path(__file__).resolve()
    parent_dir = str(current_dir.parents[1])

    annot_audio_f = '{:s}/data/speech_annotations/Despicable_Me_English.wav'.format(parent_dir)
    fs_aa, annot_audio = wavfile.read(annot_audio_f)
    annot_audio = signal.resample(annot_audio[:,0], int(round(len(annot_audio)  * (fs_audio / fs_aa))))

    # Cross-correlate the two audio streams to correct for the offset
    xcorr = signal.correlate(audio_data, annot_audio, 'same')
    lags = signal.correlation_lags(len(audio_data), len(annot_audio), 'same')
    
    n_shift = lags[np.argmax(xcorr)]
    t_shift = n_shift / fs_audio

    # Load the annotations 
    annot_file = '{:s}/data/speech_annotations/Despicable_Me_English.TextGrid'.format(parent_dir)
    
#    text_grid = textgrids.TextGrid(annot_file)
    
    # Words
#    word_container = add_text_to_module(text_grid, 'words', t_shift)
#    annot_module.add(word_container)
    
    # Phonemes
 #   phone_container = add_text_to_module(text_grid, 'phones', t_shift)
 #   annot_module.add(phone_container)
    
    #%% Write NWB again
    io.write(nwb)
    io.close()
    
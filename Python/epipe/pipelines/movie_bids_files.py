#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 13 18:08:29 2025

@author: max
"""
import numpy as np
import pandas as pd
from pynwb import NWBHDF5IO

#%% Add BIDS event and channel files
def run(nwb_fname):
    
     io = NWBHDF5IO(nwb_fname, mode='r+', load_namespaces=True)
     nwb = io.read()

     if 'task-freeviewing' in nwb_fname:
         
         if 'labels' in nwb.acquisition['TTL'].fields.keys():  
             val_img = np.where(nwb.acquisition['TTL'].labels == 'image_onset')[0]
         elif 'data__labels' in nwb.acquisition['TTL'].fields.keys():   
             val_img = np.where(nwb.acquisition['TTL'].data__labels == 'image_onset')[0]
             
         idx_img = nwb.acquisition['TTL'].data[:] == val_img
         
         t_onset = nwb.acquisition['TTL'].timestamps[idx_img]
         
         duration = nwb.trials['stop_time'][:] - nwb.trials['start_time'][:]
         
         stim_files = ['freeviewing/{:s}.png'.format(s) 
                       for s in nwb.trials['stim'][:]]
                       
         event_df = pd.DataFrame({'onset': t_onset, 
                                  'duration': duration, 
                                  'stim_file': stim_files})
         
         event_df.to_csv(nwb_fname.replace('_ieeg.nwb', '_events.tsv'), 
                         sep='\t', index=False)
      
     else:
         
         if 'task-rest' in nwb_fname:
             onset_label = 'onset'
             offset_label = 'offset'
         else:
             onset_label = 'movie_onset'
             offset_label = 'movie_offset'
         
         if 'labels' in nwb.acquisition['TTL'].fields.keys():    
             val_onset = np.where(nwb.acquisition['TTL'].labels == onset_label)[0]
         elif 'data__labels' in nwb.acquisition['TTL'].fields.keys():   
             val_onset = np.where(nwb.acquisition['TTL'].data__labels == onset_label)[0]
             
         idx_onset = nwb.acquisition['TTL'].data[:] == val_onset
         
         t_onset = nwb.acquisition['TTL'].timestamps[idx_onset][0]
         
         if 'labels' in nwb.acquisition['TTL'].fields.keys():    
             val_offset = np.where(nwb.acquisition['TTL'].labels == offset_label)[0]
         elif 'data__labels' in nwb.acquisition['TTL'].fields.keys():   
             val_offset = np.where(nwb.acquisition['TTL'].data__labels == offset_label)[0]
                     
         idx_offset = nwb.acquisition['TTL'].data[:] == val_offset
         
         t_offset = nwb.acquisition['TTL'].timestamps[idx_offset][0]
         
         duration = t_offset - t_onset
         
         if 'task-rest' in nwb_fname:
             event_df = pd.DataFrame({'onset': [t_onset], 
                                      'duration': [duration]})
         else:
             event_df = pd.DataFrame({'onset': [t_onset], 
                                      'duration': [duration], 
                                      'stim_file': nwb.stimulus['ExternalFiles'].external_file[:]})
         
         event_df.to_csv(nwb_fname.replace('_ieeg.nwb', '_events.tsv'), 
                         sep='\t', index=False)
     
     #%% Channel file
     spec = nwb.electrodes.spec[:]
     
     spec[spec == 'depth'] = 'SEEG'
     spec[spec == 'hd_depth'] = 'SEEG'
     spec[spec == 'strip'] = 'ECOG'
     
     units = np.tile('uV', len(spec))
     low_cutoff = np.tile('n/a', len(spec))
     high_cutoff = np.tile('n/a', len(spec))
     
     status = np.array(['good'] * len(spec))
     status[nwb.electrodes.artifact_patient[:] == 1] = 'bad'
     
     channel_df = pd.DataFrame({'name': nwb.electrodes.label[:],
                                'type': spec,
                                'units': units,
                                'low_cutoff': low_cutoff,
                                'high_cutoff': high_cutoff,
                                'status': status})
     
     channel_df.to_csv(nwb_fname.replace('_ieeg.nwb', '_channels.tsv'), 
                       sep='\t', index=False)

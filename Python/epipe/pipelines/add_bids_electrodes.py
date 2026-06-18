#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 13 18:15:23 2025

@author: max
"""
import os, subprocess, json
from itertools import compress
import numpy as np
import pandas as pd
import nibabel as nib
from pynwb import NWBHDF5IO

#%% Function to copy multiple files 
def copy_sub_files(source_dir, target_dir, files_include):

    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    
    sub_folder = os.path.split(source_dir)[-1]
    
    source_files = os.listdir(source_dir)
    
    if sub_folder == 'label' or sub_folder == 'surf':
        
        idx_include = [np.sum([np.sum(np.isin(['lh.{:s}'.format(si), 
                                               'rh.{:s}'.format(si)], sf)) != 0 
                               for si in files_include]) != 0 
                       for sf in source_files]
    
    else:
        idx_include = np.isin(source_files, files_include)
       
    source_files = list(compress(source_files, idx_include))

    for i,f in enumerate(source_files):   
        cmd_cp = ('cp {:s}/{:s} {:s}/{:s}').format(source_dir, f, target_dir, f)                                                                   
        returned_value = subprocess.call(cmd_cp, shell=True)
        print('Copy {:s} file {:d}/{:d}:'.format(os.path.split(source_dir)[-1], 
                                                 i+1, 
                                                 len(source_files)), 
              returned_value)

def run(subject_id, recon_dir, nwb_dir, bids_sub=None, bids_ses=None, 
        labels_include=None, surf_include=None, mri_include=None, trans_include=None,
        elec_size_mm2=20):
        
    #%% Set defaults
    if labels_include == None:
        labels_include = ['aparc.a2009s.annot', 'aparc.annot', 
                          'Yeo2011_7Networks_N1000.annot', 
                          'Yeo2011_17Networks_N1000.annot',
                          'lh.HCP-MMP1']
    if surf_include == None:
        surf_include = ['pial', 'inflated']
        
    if mri_include == None:
        mri_include = ['aparc+aseg.mgz', 'brainmask.mgz', 'orig.mgz']
    
    if trans_include == None:
        trans_include = ['talairach.xfm']

    #%% Get freesurfer version
    stamp_file = '{:s}/{:s}/scripts/build-stamp.txt'.format(recon_dir, subject_id)
    
    if os.path.exists(stamp_file):   
        file = open(stamp_file, "r")
        fs_version = file.readline()
        fs_version = fs_version.strip('\n')
        file.close()
        
    else:    
        fs_version ='freesurfer_version_unknown';
    
    #%% Create output directory 
    surf_out_dir = '{:s}/derivatives/surfaces'.format(nwb_dir)
        
    # Derivative folder for surfaces
    if bids_sub is not None and bids_ses is not None:
        sub_deriv_dir = '{:s}/{:s}/{:s}/anat'.format(surf_out_dir, bids_sub, bids_ses)
    else:
        sub_deriv_dir = '{:s}/{:s}/anat'.format(surf_out_dir, subject_id)
    
    if not os.path.exists(sub_deriv_dir):
        os.makedirs(sub_deriv_dir)
        
    # Anatomy folder for MRIs
    if bids_sub is not None and bids_ses is not None:
        sub_anat_dir = '{:s}/{:s}/{:s}/anat'.format(nwb_dir, bids_sub, bids_ses)
    else:
        sub_anat_dir = '{:s}/{:s}/anat'.format(nwb_dir, subject_id)
    
    if not os.path.exists(sub_anat_dir):
        os.makedirs(sub_anat_dir)
        
    # ieeg folder for coordinate files
    sub_ieeg_dir = sub_anat_dir.replace('anat', 'ieeg')
    
    #%% Get raw structural MRI scans
    
    #%% Filenames for pre-implant scans
    fname_pre = '{:s}/{:s}/mri/orig/001.mgz'.format(recon_dir, subject_id)
    
    if bids_sub is not None and bids_ses is not None:
        out_file_pre = '{:s}/{:s}_{:s}_acq_preimplant_T1w.nii.gz'.format(sub_anat_dir, bids_sub, bids_ses)
    else:
        out_file_pre = '{:s}/{:s}_acq_preimplant_T1w.nii.gz'.format(sub_anat_dir, subject_id)
        
    # Load the mgz file
    img = nib.load(fname_pre)
            
    # Get the data array and affine transformation
    data = img.get_fdata()
    affine = img.affine
    
    # Create a new NIfTI image
    nii_img = nib.Nifti1Image(data, affine)
    
    # Save the NIfTI image
    nib.save(nii_img, out_file_pre)
    
    # Postimplant can just be copied
    fname_post = '{:s}/{:s}/elec_recon/postimpRaw.nii.gz'.format(recon_dir, subject_id)
    
    if bids_sub is not None and bids_ses is not None:
        out_file_post = '{:s}/{:s}_{:s}_acq_postimplant_T1w.nii.gz'.format(sub_anat_dir, bids_sub, bids_ses)
    else:
        out_file_post = '{:s}/{:s}_acq_postimplant_T1w.nii.gz'.format(sub_anat_dir, subject_id)
        
    cmd_cp = ('cp {:s} {:s}').format(fname_post, out_file_post)                                                                   
    returned_value = subprocess.call(cmd_cp, shell=True)
    print('Copy postimplant MRI:', returned_value)
        
    
    #%% Get freesurfer (derivate data)
    
    # Freesurfer version
    fs_version_file = '{:s}/freesurfer_version.txt'.format(sub_deriv_dir)
    
    with open(fs_version_file, 'w') as file:
        file.write(fs_version)
    file.close()
    
    # FreeSurfer pre-processed preimplant scan
    fname_fs_pre = '{:s}/{:s}/elec_recon/T1.nii.gz'.format(recon_dir, subject_id)
    
    if bids_sub is not None and bids_ses is not None:
        out_file_pre = '{:s}/{:s}_{:s}_acq_preimplant_T1w.nii.gz'.format(sub_deriv_dir, bids_sub, bids_ses)
    else:
        out_file_pre = '{:s}/{:s}_acq_preimplant_T1w.nii.gz'.format(sub_deriv_dir, subject_id)
    
    cmd_cp = ('cp {:s} {:s}').format(fname_fs_pre, out_file_pre)                                                                   
    returned_value = subprocess.call(cmd_cp, shell=True)
    print('Copy fsaverage pre-implant MRI:', returned_value)
    
    # Postimplant scan aligned to preimplant scan
    fname_fs_post = '{:s}/{:s}/elec_recon/postInPre.nii.gz'.format(recon_dir, subject_id)
    
    if bids_sub is not None and bids_ses is not None:
        out_file_post = '{:s}/{:s}_{:s}_acq_postimplant_T1w.nii.gz'.format(sub_deriv_dir, bids_sub, bids_ses)
    else:
        out_file_post = '{:s}/{:s}_acq_postimplant_T1w.nii.gz'.format(sub_deriv_dir, subject_id)
        
    cmd_cp = ('cp {:s} {:s}').format(fname_fs_post, out_file_post)                                                                   
    returned_value = subprocess.call(cmd_cp, shell=True)
    print('Copy fsaverage aligned post-implant MRI:', returned_value)
    
    #%% Labels
    label_dir = '{:s}/{:s}/label'.format(recon_dir, subject_id)
    label_out_dir ='{:s}/label'.format(sub_deriv_dir)
    
    copy_sub_files(label_dir, label_out_dir, labels_include)
        
    #%% Surfaces
    surf_dir = '{:s}/{:s}/surf'.format(recon_dir, subject_id)
    surf_sub_dir ='{:s}/surf'.format(sub_deriv_dir)
    
    copy_sub_files(surf_dir, surf_sub_dir, surf_include)
    
    #%% MRI volumes
    mri_dir = '{:s}/{:s}/mri'.format(recon_dir, subject_id)
    mri_out_dir ='{:s}/mri'.format(sub_deriv_dir)
    
    copy_sub_files(mri_dir, mri_out_dir, mri_include)
    
    #%% MRI transforms
    trans_dir = '{:s}/{:s}/mri/transforms'.format(recon_dir, subject_id)
    trans_out_dir ='{:s}/mri/transforms'.format(sub_deriv_dir)
    
    copy_sub_files(trans_dir, trans_out_dir, trans_include)
    
    #%% Copy fsaverage if it doesn't exist
    fsaverage_dir = '{:s}/sub-fsavearge'.format(surf_out_dir)
    
    if not os.path.exists(fsaverage_dir):
        
        os.makedirs(fsaverage_dir)
        
        # Labels
        label_dir = '{:s}/fsaverage/label'.format(recon_dir)
        label_out_dir ='{:s}/label'.format(fsaverage_dir)
    
        copy_sub_files(label_dir, label_out_dir, labels_include)
            
        # Surfaces
        surf_dir = '{:s}/fsaverage/surf'.format(recon_dir)
        surf_sub_dir ='{:s}/surf'.format(fsaverage_dir)
    
        copy_sub_files(surf_dir, surf_sub_dir, surf_include)
    
        
    #%% Electrode location files 
    # https://bids-specification.readthedocs.io/en/stable/modality-specific-files/intracranial-electroencephalography.html
    
    #%% Load NWB to get these
    
    # Get first file
    nwb_files = os.listdir(sub_ieeg_dir)
    nwb_fname = '{:s}/{:s}'.format(sub_ieeg_dir, nwb_files[0])
    
    #%% fsaverage
    if bids_sub is not None and bids_ses is not None:
        coords_fname = '{:s}/{:s}_{:s}_space-fsaverage_electrodes.tsv'.format(sub_ieeg_dir, 
                                                                              bids_sub, 
                                                                              bids_ses)
    else:
        coords_fname = '{:s}/{:s}_space-fsaverage_electrodes.tsv'.format(sub_ieeg_dir, 
                                                                         subject_id)
        
    coords_json = coords_fname.replace('.tsv', '.json')
    
    io = NWBHDF5IO(nwb_fname, mode='r+', load_namespaces=True)
    nwb = io.read()
    
    elec_size = np.ones(nwb.electrodes.label.shape) * elec_size_mm2
    
    # Create a dataframe with electrode info in fsaverage space
    df_fsaverage = pd.DataFrame(
        {
         'name': nwb.electrodes.label[:],
         'x': nwb.electrodes.x[:],
         'y': nwb.electrodes.y[:],
         'z': nwb.electrodes.z[:],
         'size': elec_size,
         'group': nwb.electrodes.group_name[:], 
         'hemisphere': nwb.electrodes.hem[:]
         }
        )
    
    df_fsaverage.to_csv(coords_fname, sep='\t', index=False)
    
    # .json file
    fsaverage_dict = {
        'IntendedFor': 'derivatives/surfaces/sub-fsaverage/surf/*.pial',
        'iEEGCoordinateSystem': 'fsaverage', 
        'iEEGCoordinateUnits': 'mm', 
        'iEEGCoordinateProcessingDescription': 'yangWang-preIeegBids', 
        'iEEGCoordinateProcessingReference': 'Yang, Wang et al., 2012 NeuroImage; Groppe et al., 2017 JNeuroMeth', 
        }
    
    with open(coords_json, 'w') as outfile:
        json.dump(fsaverage_dict, outfile)
        
    #%% subject specific
    if bids_sub is not None and bids_ses is not None:
        coords_fname = '{:s}/{:s}_{:s}_space-individual_electrodes.tsv'.format(sub_ieeg_dir, 
                                                                               bids_sub, 
                                                                               bids_ses)
    else:
        coords_fname = '{:s}/{:s}_space-individual_electrodes.tsv'.format(sub_ieeg_dir, 
                                                                          subject_id)
        
    coords_json = coords_fname.replace('.tsv', '.json')
    
    # Create a dataframe with electrode info in fsaverage space
    df_fsnative = pd.DataFrame(
        {
         'name': nwb.electrodes.label[:],
         'x': nwb.electrodes.lepto_x[:],
         'y': nwb.electrodes.lepto_y[:],
         'z': nwb.electrodes.lepto_z[:],
         'size': elec_size,
         'group': nwb.electrodes.group_name[:], 
         'hemisphere': nwb.electrodes.hem[:]
         }
        )
    
    df_fsnative.to_csv(coords_fname, sep='\t', index=False)
    
    # .json file
    if bids_sub is not None and bids_ses is not None:
        surf_path = 'derivatives/surfaces/{:s}/{:s}/anat/surf/*.pial'.format(bids_sub, bids_ses)
    else:
        surf_path = 'derivatives/surfaces/{:s}/anat/surf/*.pial'.format(subject_id)
    
    fsnative_dict = {
        'IntendedFor': surf_path,
        'iEEGCoordinateSystem': 'fsnative', 
        'iEEGCoordinateUnits': 'mm', 
        'iEEGCoordinateProcessingDescription': 'yangWang-preIeegBids', 
        'iEEGCoordinateProcessingReference': 'Yang, Wang et al., 2012 NeuroImage; Groppe et al., 2017 JNeuroMeth', 
        }
    
    with open(coords_json, 'w') as outfile:
        json.dump(fsnative_dict, outfile)
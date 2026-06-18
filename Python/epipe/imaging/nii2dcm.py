#!/usr/bin/env python
# coding: utf-8
"""
This script is used to convert niftis back into dicom format by using the
original T1w dicom that the nifti is derived from. Primarily used for
conversion of T1w niftis with functional overlay back into dicom format
for usage in presurgical planning

Noah Markowitz
Human Brain Mapping Lab
North Shore University Hospital
October 2019

Original code by Joo-won Kim
"""

import pydicom as dicom
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import argparse
import datetime
import sys
import os
import glob

def main(niifile, orig_dicom_dir,dcm_output_dir,studyID='HBML_nii2dcm'):

    # Check that input exists and is of correct type
    if not os.path.isfile(niifile):
        print('Sorry! %s is not an existing file!' % niifile)
    else:
        imgName = os.path.basename(niifile).split('.gz')[0].split('.nii')[0]

    if not os.path.isdir(orig_dicom_dir):
        print('Sorry! %s is not an existing directory!' % orig_dicom_dir)

    if not dcm_output_dir:
        dcm_output_dir = niifile.split('.gz')[0].split('.nii')[0]

    while os.path.isdir(dcm_output_dir):
        dcm_output_dir += '+'

    # If studyID is a list then make it a string
    # if isinstance(studyID, list): studyID = ' '.join(studyID)
    # Today's date
    ProcessingDay = datetime.date.today().strftime("%Y%m%d")

    # Create the general output directory if it doesn't exist
    if not os.path.isdir(dcm_output_dir):
        os.mkdir(dcm_output_dir)

    # Generate randomization seed
    N = 0
    for x in niifile:
        N = N + ord(x)

    # Set seed
    np.random.seed(N)

    # The series number that will be assigned to this dicom
    numD = np.random.randint(1,1000,1).astype(int)[0]

    print('\n\nNow Starting to convert nifti to dicoms')

    # print(dcm_output_dir)

    # Read in nii file
    img = nib.load(niifile)
    dat_seg = img.get_fdata()
    dat_seg = dat_seg.astype('uint16') # THIS SHOULD BE THE SAME DATATYPE AS T1W

    # Read DICOM orig
    lst_dicom = glob.glob(os.path.join(orig_dicom_dir, '*.dcm'))

    # Set some variables beforehand for the new dicom to be made
    sample_dcm = dicom.read_file(lst_dicom[0])
    accss_num = sample_dcm[('0008', '0050')].value
    study_desc = sample_dcm[('0008', '1030')].value
    study_id = sample_dcm[('0020', '0010')].value
    study_instance_uid = sample_dcm[('0020', '000d')].value.split('.')
    series_instance_uid = sample_dcm[('0020', '000e')].value.split('.')
    series_description = sample_dcm[('0008', '103e')].value
    uid = sample_dcm[('0008', '0018')].value.split('.')
    study_date = sample_dcm[('0008','0020')].value

    # Change some of the StudyID metadata
    accss_num = '9' * len(accss_num)
    study_desc = studyID
    study_id = '9' * len(study_id)
    study_instance_uid[-1] = '7' * len(study_instance_uid[-1])
    study_instance_uid = '.'.join(study_instance_uid)

    # Patient ID
    patID = sample_dcm[('0010', '0020')].value
    newPatID = ''.join(np.random.randint(0, 9, len(patID)).astype('str').tolist())

    # Create new series instance uid
    l = len(series_instance_uid[-4])
    n = np.random.randint(0, 9, l).astype(str).tolist()
    series_instance_uid[-4] = ''.join(n[:])
    series_instance_uid = '.'.join(series_instance_uid)

    # Create new uid
    uid_num = np.random.randint(0,99,1).astype(int)[0]

    # How to slice the nifti to match dicoms
    ndicoms = len(lst_dicom)
    nrows = sample_dcm[('0028', '0010')].value
    ncolumns = sample_dcm[('0028', '0011')].value
    slice_acq = img.shape.index(ndicoms)
    if slice_acq == 0:
        def get_slice(nii_data, slice_num):
            return nii_data[slice_num, :, :]
    elif slice_acq == 1:
        def get_slice(nii_data, slice_num):
            return nii_data[:, slice_num, :]
    elif slice_acq == 2:
        def get_slice(nii_data, slice_num):
            return nii_data[:, :, slice_num]
    else:
        print('Whoops! Something is wrong')

    # generate DICOM outputs
    for fn in lst_dicom:
        dcm = dicom.read_file(fn)

        # get dicom values
        uid = dcm[('0008', '0018')].value.split('.')
        uid_orig = dcm[('0008', '0018')].value.split('.')
        series_description = dcm[('0008', '103e')].value
        slc = int(dcm[('0020', '0013')].value)
        # orientation = dcm[('0020', '0037')].value

        dat = dcm.pixel_array

        # NIFTI slice order is reversed
        slc_nifti = slc - 1
        dat_seg_slc = np.array(get_slice(dat_seg, slc_nifti).T)
        dat_seg_slc = np.fliplr( np.flipud(dat_seg_slc) )

        # Uncomment this to double check orientation is now correct
        # fig,ax = plt.subplots(1,2)
        # ax[0].imshow(dat, cmap='gray', origin='lower')
        # ax[0].set_title('original dicom')
        # ax[1].imshow(dat_seg_slc, cmap='gray', origin='lower')
        # ax[1].set_title('nifti image')

        # create new uid, series
        # create new UID not to confuse DICOM reader
        uid[-1] = uid_orig[-1][:8] + str(uid_num).zfill(2) + uid_orig[-1][8:]
        dcm[('0008', '0018')].value = '.'.join(uid)
        dcm[('0010', '0010')].value = 'HBML^Proc'
        series_new = series_description + '_' + imgName
        dcm[('0020', '0011')].value = numD
        dcm[('0008', '103e')].value = series_new
        dcm[('0018', '1030')].value = series_new
        dcm[('7fe0', '0010')].value = dat_seg_slc.tobytes()
        dcm[('0008', '0050')].value = accss_num
        dcm[('0008', '1030')].value = study_desc
        dcm[('0020', '0010')].value = study_id
        dcm[('0020', '000d')].value = study_instance_uid
        dcm[('0020', '000e')].value = series_instance_uid

        try:
            dcm[('0008', '0012')].value = ProcessingDay
        except:
            x = 2 + 2

        dcm[('0008', '0020')].value = ProcessingDay
        dcm[('0008', '0021')].value = ProcessingDay
        dcm[('0008', '0022')].value = ProcessingDay
        dcm[('0008', '0023')].value = ProcessingDay
        dcm[('0010', '0020')].value = newPatID
        dcm[('0010','0030')].value = '19000101'
        # dcm[('0020', '0037')].value = new_orientation
        # fn_out = 'MR.' + dcm[('0008', '0018')].value + '.dcm'
        fn_out = os.path.basename(fn)
        dcm.save_as(os.path.join(dcm_output_dir, fn_out))

def cmnd_line_parser():
    # Create argument parser
    parser = argparse.ArgumentParser(
        description='For converting nifti images back into dicom after being overlayed with binary fMRI activation masks',
        prog='nii2dcm2.py', usage='%(prog)s -in T1_w_bin_activation.nii.gz -o processed_dicoms/ -d T1w_dicom_dir/')
    parser.add_argument('-i', '-in', nargs='*', help='A single nifti to be converted to dicoms',
                        required=True, dest='niifile', type=str)
    parser.add_argument('-o', '-out', nargs=1, help='Directory to place converted dicoms into', required=False, dest='dcm_output_dir', type=str)
    parser.add_argument('-d', '-dcm', nargs=1, help='Original T1 dicom to use as the template for conversion', required=True,
                        dest='orig_dicom', type=str)
    parser.add_argument('-s', '-study', nargs=1, help='Study ID that will be shown in Dicom header (Default is \'HBML Processing\')',
                        required=False, default='HBML_nii2dcm', dest='studyID', type=str)


    # Parse arguments
    args = parser.parse_args()
    if args.dcm_output_dir is None:
        main(args.niifile[0], args.orig_dicom[0],studyID=args.studyID)
    else:
        main(args.niifile[0], args.orig_dicom[0], args.dcm_output_dir[0], studyID=args.studyID)

if __name__ == '__main__':
    cmnd_line_parser()


import numpy as np
from pyedflib.highlevel import read_edf_header
from pyedflib import EdfReader, EdfWriter
from pyedflib import FILETYPE_BDF, FILETYPE_BDFPLUS, FILETYPE_EDF, FILETYPE_EDFPLUS
from mne.io import read_raw_edf
from datetime import datetime, timezone, timedelta
from dateutil.tz import tzlocal
import re
from epipe import load_nwb_settings

def _stamp_to_dt(utc_stamp):
    """Convert timestamp to datetime object in Windows-friendly way."""
    if 'datetime' in str(type(utc_stamp)): return utc_stamp
    # The min on windows is 86400
    stamp = [int(s) for s in utc_stamp]
    if len(stamp) == 1:  # In case there is no microseconds information
        stamp.append(0)
    return (datetime.fromtimestamp(0, tz=timezone.utc) +
            timedelta(0, stamp[0], stamp[1]))  # day, sec, μs


# TODO: For anonymization, replace there:
#   * Patient name
#   * dob
#   * date of session
#   * Patient code
#   * Technician
#   * annotations with patient info

# Check mne edf style file
#mne_edf = read_raw_edf(fname)
#mne_edf.info['meas_date']
def _pyedflib_anon(edf_fname,new_edf_fname=None):

    if new_edf_fname == None:
        new_edf_fname = edf_fname[:-4] + '_anon.edf'

    # pyedflib read edf
    hdr = read_edf_header(edf_fname)
    raw_edf = EdfReader(edf_fname)
    hbml_settings = load_nwb_settings()

    #region Get and set some info
    nchans = len(raw_edf.getSignalHeaders())
    signal_headers = raw_edf.getSignalHeaders()
    dig_max = raw_edf.getDigitalMaximum()
    dig_min = raw_edf.getDigitalMinimum()
    phys_max = raw_edf.getPhysicalMaximum()
    phys_min = raw_edf.getPhysicalMinimum()
    new_pt_name = 'Business,Nunya'
    birth_date = datetime(1900,1,1,tzinfo=timezone.utc)
    new_pt_code = 'X'
    new_technician = 'noah-markowitz-hbml-manhasset'
    file_start_time = datetime.strptime(hbml_settings['meta_data']['date'],"%Y-%m-%d %H:%M:%S").replace(tzinfo=tzlocal())

    # Filetype
    if raw_edf.filetype == FILETYPE_EDFPLUS:
        ftype = FILETYPE_EDFPLUS
    else:
        ftype = FILETYPE_EDF

    # Annotations are a numpy array containing 3 nested numpy arrays
    # The number of elements in all is the same
    # The third array contains the text of each annotation
    annotations = raw_edf.readAnnotations()
    annot_descriptions = annotations[2]
    n_annots = len(annot_descriptions)

    #endregion

    #region Create new edf

    # Init new edf object
    anon_edf = EdfWriter(new_edf_fname,n_channels=nchans,file_type=ftype)

    # Set header information
    anon_edf.setPatientCode(new_pt_code)
    anon_edf.setPatientName(new_pt_name)
    anon_edf.setTechnician(new_technician)
    anon_edf.setSignalHeaders(signal_headers)
    anon_edf.setStartdatetime(file_start_time)
    anon_edf.setBirthdate(birth_date)

    # Write signals to the file
    # WRITE_DIG = False
    # for chnii in range(nchans):
    #
    #     if WRITE_DIG:
    #         chn_data = edf.readSignal(chnii, digital=True)
    #         anon_edf.writeDigitalSamples(chn_data)
    #     else:
    #         chn_data = edf.readSignal(chnii)
    #         anon_edf.writePhysicalSamples(chn_data)
    #
    #     del chn_data

    # Create empty array
    nsamples = raw_edf.getNSamples()
    data = np.zeros((nchans,nsamples[0])).astype(np.int32)

    for chnii in range(nchans):
        chn_data = raw_edf.readSignal(chnii, digital=True)
        data[chnii,chn_data]
        del chn_data

    anon_edf.writeSamples(data,digital=True)

    #endregion

    #region Write annotations to file
    annots2ignore = hbml_settings['raw']['edf']['annotations_to_ignore']
    annotRegStr = '(?:% s)' % '|'.join(annots2ignore)
    for ii in range(n_annots):
        desc = annot_descriptions[ii]
        if not re.search(annotRegStr, desc):
            # add annotation
            onset = annotations[0][ii]
            duration = annotations[1][ii]
            desc = annotations[2][ii]
            anon_edf.writeAnnotation(onset, duration, desc)

    #endregion

    raw_edf.close()
    anon_edf.close()

def edf_anon(edf_fname,new_edf_fname=None):
    if new_edf_fname == None:
        new_edf_fname = edf_fname[:-4] + '_anon.edf'

    try:
        print('-------> Anonymizing using pyedflib library')
        _pyedflib_anon(edf_fname,new_edf_fname=new_edf_fname)
    except:
        print('-------> Error occurred while using pyedflib. Data may be discontiguous. Please check anonymized file data compared to non-anonymized file data')
        from .save_edf import write_mne_edf

        edf = read_raw_edf(edf_fname,preload=True)
        edf.anonymize()

        # Start and end to clip in seconds
        tstart = 0
        tend = 1

        # Now go through annotations and erase the ones not needed
        hbml_settings = load_nwb_settings()
        annots2ignore = hbml_settings['raw']['edf']['annotations_to_ignore']
        annotRegStr = '(?:% s)' % '|'.join(annots2ignore)
        annotations = edf.annotations
        boolIdx = []

        for a in annotations:
            if not re.search(annotRegStr, a['description']):
                boolIdx.append(True)
            else:
                boolIdx.append(False)

        new_annots = annotations[boolIdx]
        if len(new_annots) == 0:
            new_annots.append(tstart, 0.0, 'nothing here')

        edf.set_annotations(new_annots)

        write_mne_edf(edf, new_edf_fname)

def cmnd_line_parser():
    import argparse
    parser = argparse.ArgumentParser(
        description='Easily anonymize edf files. Even those with PHI in the annotations',
        prog='anonymize.py', usage='anon_edf -i my_file.edf -o my_file_anon.edf')
    parser.add_argument('-i', '-in', nargs=1, help='edf file to anonymize',
                        required=True, dest='raw_edf', type=str)
    parser.add_argument('-o', '-out', nargs=1,
                        help='New edf name. If none given then will create a new file appended with "_anon.edf"',
                        required=False, dest='new_edf',
                        type=str,default=None)

    args = vars(parser.parse_args())
    print(args['raw_edf'])
    print(args)
    print(args['new_edf'])
    raw_edf_fname = args['raw_edf'] if isinstance(args['raw_edf'],str) else args['raw_edf'][0]
    new_edf_fname = args['new_edf'][0] if isinstance(args['new_edf'],list) else args['new_edf']
    edf_anon(raw_edf_fname,new_edf_fname=new_edf_fname)

if __name__ == '__main__':
    cmnd_line_parser()
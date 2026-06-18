import numpy as np
from mne.io import RawArray
from mne import create_info, Annotations
from mne.channels import make_dig_montage
from pynwb import TimeSeries, NWBFile
from pynwb.ecephys import ElectricalSeries
from epipe.fileio.nwb.utils import load_nwb_settings
from .utils import inspectNwb

# TODO: Allow nwb2mne to pass in freesurfer directory with elec_recon to create montage and specify coordinates (LEPTO or fsaverage)

def _getRoot(nwbDataInterface):
    """Simple function to get the root of a nwb file when a child is passed in"""
    c = nwbDataInterface
    #parent = nwbDataInterface
    ii = 0
    while not isinstance(c,NWBFile):
        ii += 1
        #parent = c
        #c = parent.get_ancestor()
        c = c.get_ancestor()
        if ii >= 1000:
            print('Error in code')
            break

    return c

def nwb2mne(id,nwbfile=None,preload=False, create_montage=True):
    """Function for converting a TimeSeries object to a MNE RawArray object

    Parameters
    ----------
    id : TimeSeries, str
        A TimeSeries object, name or identifier of a TimeSeries object (latter two require the nwbfile argument)
    nwbfile : NWBFile
        The root object. Only needed if id is a str
    preload : bool
        Whether to load all the data

    Returns : RawArray
        An instance of mne.io.RawArray
    -------

    """

    # Get the TimeSeries instance
    if isinstance(id,TimeSeries):
        ts = id
        nwbfile = _getRoot(ts)
    elif isinstance(id,str) and isinstance(nwbfile,NWBFile):
        nwbObjIds = list(nwbfile.objects.keys())
        if id in nwbObjIds:
            ts = nwbfile.objects.get(id)
        else:
            inspection = inspectNwb(nwbfile)
            ts_df = inspection['timeseries']
            if id in ts_df['name'].values:
                objId = ts_df.loc[ts_df['name'] == id, 'id'].values[0]
                ts = nwbfile.objects.get(objId)
            else:
                print('ERROR!')
    else:
        print('ERROR!')

    # Set chan type for MNE
    if ts.name == 'ieeg':
        ch_types = 'seeg'
    elif isinstance(ts,ElectricalSeries):
        ch_types = 'eeg'
    else:
        ch_types = 'misc'

    # Determine sampling rate
    fs = ts.rate if 'rate' in ts.fields.keys() else 'timestamps'

    # Get number of channels and load data
    if len(ts.data.shape) == 2:
        ntpts, nchans = ts.data.shape
    else:
        ntpts = ts.data.size
        nchans = 1

    # Get timepoints
    if fs == 'timestamps':
        tvec = ts.timestamps[()]
        rate = 1/np.median(np.diff(tvec))
        t1 = tvec[0] + ts.starting_time
        #tvec_idx = np.logical_and(tvec>=t1, tvec<=t2)
    elif isinstance(fs,float) or isinstance(fs,int):
        rate = ts.rate
        tvec = np.arange(0,ntpts)/rate
        t1 = tvec[0] + ts.starting_time
        #tvec_idx = np.logical_and(tvec>=t1, tvec<=t2)

    # Get data
    if len(ts.data.shape) == 1:
        data = np.reshape(ts.data[()],(-1,ntpts))
    elif len(ts.data.shape) == 2:
        data = ts.data[()].T
    else:
        print('ERROR!')

    # Set time offset if first sample wasn't taken at time 0
    starting_time = ts.starting_time
    first_samp_offset = int(starting_time * rate)

    # Handle based on type
    if isinstance(ts,ElectricalSeries):

        # Get electrode info
        elecTable = nwbfile.electrodes.to_dataframe()
        ch_idx = ts.electrodes.data[()]
        elecs = elecTable.iloc[ch_idx]

        if 'label' in elecTable.columns:
            ch_names = elecs['label'].to_list()
        else:
            ch_names = ['ch{}'.format(ii+1) for ii in range(len(ch_idx))]

        # Create Info and Raw objects
        info = create_info(ch_names,sfreq=rate,ch_types=ch_types)
        info['line_freq'] = 60
        info['description'] = nwbfile.experiment_description
        raw = RawArray(data, info, first_samp=first_samp_offset)

        # Get bad channels
        annots, bad_chans = _badChanAnnotation(ts,raw.times[-1],return_chans=True)
        #raw.set_annotations(annots)
        raw.info['bads'] = bad_chans

        # Now add the montage
        if all([c in list(elecTable.columns) for c in ["lepto_x", "lepto_y", "lepto_z"]]):
            xyz_cols = ["lepto_x", "lepto_y", "lepto_z"]
        else:
            xyz_cols = ["x","y","z"]

        ch_coords = {}
        for idx, row in elecTable.iterrows():
            ch_name = row["label"]
            if ch_name in raw.ch_names:
                xyz = row[xyz_cols].to_numpy() / 1000
                ch_coords[ch_name] = xyz

        montage = make_dig_montage(ch_pos=ch_coords, coord_frame='mri')
        if create_montage:
            raw.set_montage(montage, on_missing="ignore")
        #raw.set_montage(montage)


    else:
        ch_names = ['ch{}'.format(ii+1) for ii in range(nchans)]
        info = create_info(ch_names,sfreq=rate,ch_types=ch_types)
        info['line_freq'] = 60
        info['description'] = nwbfile.experiment_description
        raw = RawArray(data, info, first_samp=first_samp_offset)

    return raw

def _badChanAnnotation(es,end_time,return_chans=False):

    # Load NWB settings
    nwbSettings = load_nwb_settings()
    bad_chan_params = nwbSettings['ecog_file']['fields']
    elecTable = es.electrodes.to_dataframe()

    all_chans = []

    # Initiate Annotations object
    annots = None
    chan_types = list(bad_chan_params.keys())
    ntypes = len(chan_types)

    # Make annotations object and get channels with tags
    for k in bad_chan_params.keys():
        cols = bad_chan_params[k]
        bad_chans = []
        for c in cols:
            if c in list(elecTable.columns):
                chans = elecTable.loc[elecTable[c] == 1,'label'].tolist()
                bad_chans += chans
                all_chans += chans

        bad_chans = np.unique( np.array(bad_chans) ).tolist()
        if annots == None:
            annots = Annotations(0,end_time,k,ch_names=[bad_chans])
        elif isinstance(annots,Annotations):
            annots.append(0,end_time,k,ch_names=[bad_chans])

    if return_chans:
        all_chans = np.unique(np.array(all_chans)).tolist()
        return annots, all_chans
    else:
        return annots


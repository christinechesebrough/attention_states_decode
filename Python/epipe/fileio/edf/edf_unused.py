from datetime import datetime
from dateutil.tz import tzlocal
import re
from mne.io import read_raw_edf
from pyedflib import FILETYPE_EDFPLUS
from pyedflib.highlevel import read_edf_header, read_edf
from pyedflib import EdfReader, EdfWriter
from epipe.fileio.edf.save_edf import write_mne_edf
from epipe.fileio.edf.mne_extras import write_edf as write_mne_edf_extras
from epipe.fileio.nwb.utils import load_nwb_settings

nwb_settings = load_nwb_settings()
annots2ignore = nwb_settings['raw']['edf']['annotations_to_ignore']
regstr = '(?:% s)' % '|'.join(annots2ignore)
new_start_time = datetime.strptime(nwb_settings['meta_data']['date'],"%Y-%m-%d %H:%M:%S").replace(tzinfo=tzlocal())
fname = r"C:\Users\nmarkowitz\Documents\SOURCEDATA\edf_anonymize_practice\NS134_SZ2.edf"

#region pyedflib funcs
hdr = read_edf_header(fname,True) # Just the header and annotations
hilvl_read = read_edf(fname) # Tuple of (neural_data, hdr, annotations)
del hilvl_read
edf = EdfReader(fname)
new_fname = fname.replace('.edf','_anon3.edf')

f = EdfWriter(new_fname,len(hdr['channels']),file_type=FILETYPE_EDFPLUS)



#endregion

#region mne save_edf from skjerns
edf = read_raw_edf(sample_edf)
edf.close()

#endregion

#region mne save_edf from mne_extras
edf = read_raw_edf(sample_edf,preload=True)
edf.anonymize()
# edf.info['meas_date'] = datetime.now()
# edf._raw_extras[0]['subject_info'] = {
#     'id': 'X',
#     'sex': 'U',
#     'birthday': new_start_time,
#     'name': 'MisterNoah'
# }

# Start and end to clip in seconds
tstart = 0
tend = 1

# Now go through annotations and erase the ones not needed
annotations = edf.annotations
boolIdx = []
for a in annotations:

    no_phi = re.search(regstr, a['description']) == None
    within_time = a['onset'] >= tstart and a['onset'] <= tend
    if no_phi and within_time:
        boolIdx.append(True)
    else:
        boolIdx.append(False)

# New annotations to use
a2 = annotations[boolIdx]
if len(a2) == 0:
    a2.append(tstart,0.0,'nothing here')

edf.set_annotations(a2)

# Set where to save the file
new_fname = sample_edf.replace('.edf','_anon.edf')
new_fname2 = sample_edf.replace('.edf','_anon2.edf')

# Write out the file
iswritten = write_mne_edf_extras(edf,new_fname)
iswritten = write_mne_edf(edf,new_fname2)
iswritten = write_mne_edf(edf,new_fname,tmin=tstart,tmax=tend)

#endregion
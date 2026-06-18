# NWB IO functions
from .fileio.nwb.ieeg2nwb import ieeg2nwb
from .fileio.nwb.utils import load_nwb_settings, compress_data, inspectNwb
from .fileio.nwb.nwb2mne import nwb2mne
from .fileio import load_hbml_data

# Other IO Functions
from .fileio import read_ecog, read_mgrid, read_ielvis, read_presentation_log
from .fileio.tdt import read_tdt_ttls, getTDTStore

# Imaging
from .imaging.nii2dcm import main as nii2dcm

# iEEG Preprocessing functions
from .preproc_ieeg.events import ana2dig, create_toi_slice
from .preproc_ieeg.signal_processing import filter_hfa_continuous, filter_hfa_epochs
from.preproc_ieeg.channels import reref_avg, reref_bipolar, reref_white_matter,reref_white_matter_strict

# Pipelines
# from .pipelines import et2nwb
from .pipelines import visual_localizer, preprocess_movies
from .pipelines import visual_localizer

# Plotting tools
from .plotters import plot_ortho, plot_brain_surf, plot_stem_annotation

# Misc
from .utilities import create_mne_report_table, find_nearest_vertex, mne_tfr, recurse_np2list, get_ptd_index

from .data import read_aseg_csv



# Create the __all__ list for documentation
import types
__all__ = [
    'ieeg2nwb',
    'load_nwb_settings',
    'compress_data',
    'inspectNwb',
    'nwb2mne',
    'load_hbml_data',
    'read_ecog',
    'read_mgrid',
    'read_ielvis',
    'read_presentation_log',
    'read_tdt_ttls',
    'getTDTStore',
    'nii2dcm',
    'ana2dig',
    'create_toi_slice',
    'filter_hfa_continuous',
    'filter_hfa_epochs',
    'reref_avg',
    'reref_bipolar',
    'reref_white_matter',
    'plot_ortho',
    'plot_brain_surf',
    'plot_stem_annotation',
    'create_mne_report_table',
    'create_indiv_mapping',
    'find_nearest_vertex',
    'mne_tfr',
    'recurse_np2list',
    'visual_localizer'
]
"""
Noah Markowitz
"""
import numpy as np
from tqdm import tqdm
import os
from nibabel.freesurfer.io import read_geometry, read_annot, write_annot
from joblib import Parallel, delayed

# Shorthands for the different atlases as part of freesurfer"
ATLASES = {
    'dk': 'aparc',
    'd': 'aparc.a2009s',
    'hcp': 'HCP-MMP1',
    'y7': 'Yeo2011_7Networks_N1000',
    'y17': 'Yeo2011_17Networks_N1000'
}

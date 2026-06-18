from pymatreader import read_mat
import pandas as pd
import os.path as op
from ..data import _get_data_directory

def load_hbml_data():

    """Load HBML electrode table into memory"""

    fname = op.join(_get_data_directory(), 'ElecTable', 'HBML_DATA.mat')

    hbml_data = read_mat(fname)

    # Make pandas DataFrame
    df = pd.DataFrame(hbml_data['HBML_DATA'])

    # Convert some columns to boolean
    bool_cols = ["soz", "spikey", "out", "bad", "complete_info"]
    for c in bool_cols:
        df[c] = df[c].astype(bool)

    return df
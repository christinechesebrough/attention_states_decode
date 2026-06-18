def _get_data_directory():
    import os.path as op
    return op.join(op.dirname(op.dirname(op.dirname(__file__))), 'Data')

def read_aseg_csv():
    import pandas as pd
    import os.path as op
    aseg_csv = op.join(_get_data_directory(), 'aseg.csv')
    return pd.read_csv(aseg_csv)
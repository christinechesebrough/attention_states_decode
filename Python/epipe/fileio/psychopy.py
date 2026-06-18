import pandas as pd

def read_psychopy_log(fname,keywords=None):
    log = pd.read_csv(fname,sep='\t',comment='#', header=None, names=['time','type','text'])
    return log

"""
These functions pertain to "Neuro Behavioral Systems" and any associated
files and software such as Presentation

NeuroBehavioralSystems - https://www.neurobs.com/
Presentation - https://www.neurobs.com/menu_presentation/menu_features/features_overview

"""

import pandas as pd
from datetime import datetime

def read_presentation_log(logfname):
    """

    Parameters
    ----------
    logfname - Name of the presentation logfile output from experiment

    Returns
    -------
    pandas.DataFrame

    """
    # Raw text data
    with open(logfname) as f:
        raw_text = f.read()
        raw = raw_text.splitlines()

    # Scenario name and date of experiment
    scenario = raw[0].replace('Scenario - ', '')
    exp_date = raw[1].replace('Logfile written - ', '')
    datetime_object = datetime.strptime(exp_date, '%m/%d/%Y %H:%M:%S')

    # Headers for columns
    headers = raw[3].split('\t')
    headers = [h.replace(' ', '_') for h in headers]

    # Convert the data into a list of lists
    data = []
    raw_data = raw[5:]
    for thisRow in raw_data:
        row = thisRow.split('\t')
        row = [r.replace(' ', '') for r in row]
        data.append(row)

    # Data becomes a pandas DataFrame
    # Remove any excess column headers if necessary
    df = pd.DataFrame.from_records(data)
    if df.shape[1] != len(headers):
        ndiff = df.shape[1] - len(headers)
        current_column_names = list(df.columns)
        cols2drop = current_column_names[-ndiff:]
        df = df.drop(labels=cols2drop, axis=1)

    df.columns = headers

    # If there's a cell that's only whitespace, replace it with None
    df = df.replace(to_replace='', value=None)

    # Try to convert columns to float if possible
    for c in list(df.columns):
        try:
            coldata = df[c].astype(float)
            df[c] = coldata
        except:
            pass

    # Add scenario and date as columns
    df['scenario'] = scenario
    df['date'] = datetime_object

    return df

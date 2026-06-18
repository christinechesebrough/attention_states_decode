import os
import os.path as op
import numpy as np
from tqdm import tqdm
from joblib import Parallel, delayed
from nibabel.freesurfer.io import read_geometry
import pandas as pd
from typing import Union
from ..fileio import read_ielvis


def pial_to_inflated(subject: str, subjects_dir: str=None, coords:np.array=None, labels:Union[list,np.array]=None, hem:Union[list,np.array]=None, write_to_file:bool=True, n_jobs:int=-1) -> np.array:
    """
    Convert coordinates from pial surface to inflated surface.

    Parameters
    ----------
    subject : str
        Subject ID.
    subjects_dir : str, optional
        Freesurfer subject directory. If not provided, it will be read from the MNE config file.
    coords : np.array, optional
        Input coordinates. If not provided, it will be read from the .PIAL file.
    labels : Union[list,np.array], optional
        Names of electrodes. Must be specified if coords are passed in
    hem : Union[list,np.array], optional
        Hemisphere of each electrode, Must be specified if coords are passed in
    write_to_file : bool, optional
        Whether to write out the information as subject.INF in elec_recon.
    n_jobs: int, optional
        Number of parallel jobs to run, default is -1

    Returns
    -------
    np.array
        Inflated coordinates.

    Notes
    -----
    - If `subjects_dir` is not provided, it will be read from the MNE config file.
    - If `coords` is not provided, it will be read from the .PIAL file in the subject's directory.
    - If `labels` is not provided, it will be read from the electrode data in the subject's directory.
    - The function converts the coordinates from the pial surface to the inflated surface using the nearest vertex mapping.
    - The inflated coordinates are returned as a numpy array.
    - If `write_to_file` is True, the inflated coordinates will be written out to a file named subject.INF in the elec_recon directory of the subject's directory.

    """

    if subjects_dir is None:
        from mne import get_config
        subjects_dir = get_config()['SUBJECTS_DIR']

    # If coordinates not specified then plot subject using iELVis data
    if coords is None:
        elecs_df = read_ielvis( op.join(subjects_dir, subject) )
        coords = np.array(elecs_df['PIAL'].to_list())
        labels = elecs_df['label'].to_list()
        hem = [h.lower() for h in elecs_df['hem'].to_list()]
    else:
        if labels is None:
            raise ValueError("labels must not be None, must be the same length as coords")
        elif len(labels) != coords.shape[0]:
            raise ValueError("len(labels) must equal coords.shape[0]")
        elif hem is None:
            raise ValueError("hem must not be none, must be the same length as coords")
        elif len(hem) != coords.shape[0]:
            raise ValueError("len(hem) must equal coords.shape[0]")

    # Convert to inflated coordinates
    df = find_nearest_vertex(subject, subjects_dir=subjects_dir, coords=coords, hem=hem, labels=labels, n_jobs=n_jobs)

    # Load data
    surf_dir = op.join(subjects_dir, subject, 'surf')
    lh_pial_file = surf_dir + os.sep + 'lh.inflated'
    rh_pial_file = surf_dir + os.sep + 'rh.inflated'
    verts = {}
    verts['l'], _ = read_geometry(lh_pial_file)
    verts['r'], _ = read_geometry(rh_pial_file)

    inf_coords = np.zeros((len(labels),3))

    # Iterate over each electrode to get coordinates
    for i, row in df.iterrows():
        inf_coords[i,:] = verts[row["hem"]][row["closest_vert"]]

    # Write out to file
    if write_to_file:
        from .misc import timenow
        fname=op.join(subjects_dir, subject, "elec_recon", subject + "_2.INF")
        with open(fname, 'w') as file:
            file.write(timenow() + '\n')
            file.write("R A S \n")
            np.savetxt(file, inf_coords, fmt='%.6f', delimiter=' ')

    return inf_coords

    

def find_nearest_vertex(subject, subjects_dir=None, surf="pial", coords=None, hem=None, labels=None, n_jobs=-1):
    """
    Find the nearest vertex on the cortical surface to a given set of coordinates.

    Parameters:
    ----------
    subject : str
        The name of the subject.
    subjects_dir : str, optional
        The directory where the subject's data is stored. If not provided, it will be obtained from the MNE configuration.
    coords : array-like, shape (n, 3), optional
        The coordinates of the points for which to find the nearest vertex. If not provided, it will be assumed that the coordinates are already defined.
    hem : str or list, optional
        The hemisphere(s) to consider. If 'l' or 'lh', only the left hemisphere will be considered. If 'r' or 'rh', only the right hemisphere will be considered. If not provided, both hemispheres will be considered.
    labels : array-like, shape (n,), optional
        The labels associated with each coordinate. If not provided, the labels will be assigned as consecutive integers starting from 0.
    n_jobs: int, optional
        Number of parallel jobs to run, default is -1

    Returns:
    -------
    df : pandas DataFrame
        A DataFrame containing the following columns:
        - 'label': The label associated with each coordinate.
        - 'distance': The distance between each coordinate and its nearest vertex.
        - 'closest_vert': The index of the closest vertex for each coordinate.
        - 'hem': The hemisphere associated with each label.

    """
    if subjects_dir is None:
        from mne import get_config
        subjects_dir = get_config()['SUBJECTS_DIR']

    n_elecs = coords.shape[0]

    # Interpret hemispheres
    if hem == 'l' or hem == 'lh':
        hem = ['l' for ii in range(n_elecs)]
    elif hem == 'r' or hem == 'rh':
        hem = ['r' for ii in range(n_elecs)]

    # Load data
    surf_dir = os.path.join(subjects_dir, subject, 'surf')
    lh_surf_file = surf_dir + os.sep + 'lh.' + surf
    rh_surf_file = surf_dir + os.sep + 'rh.' + surf
    verts = {}
    verts['l'], _ = read_geometry(lh_surf_file)
    verts['r'], _ = read_geometry(rh_surf_file)


    if labels is None:
        labels = np.arange(0, n_elecs, 1)

    # Function to process each coordinate
    def process_coordinate(ii):
        h = hem[ii]
        dist = np.sqrt(np.sum((verts[h] - coords[ii, :]) ** 2, axis=1))
        closest_vert = dist.argmin()
        return {'label': labels[ii], 'distance': dist.min(), 'closest_vert': closest_vert, 'hem': h}

    # Parallel processing of coordinates
    results = Parallel(n_jobs=n_jobs)(delayed(process_coordinate)(ii) for ii in tqdm(range(n_elecs), desc="Finding nearest vertices on %s" % surf, unit=' vertices', position=0, leave=True) )

    # Convert results to DataFrame
    df = pd.DataFrame(results)

    return df


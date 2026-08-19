#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 13 14:44:49 2026

@author: christine
"""


import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

import mne
import os, re,sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind
import seaborn as sns  
from statsmodels.stats.multitest import fdrcorrection
import matplotlib.patches as patches
from scipy.spatial.distance import norm
from scipy.stats import ttest_1samp,  ttest_ind
from matplotlib.colors import Normalize, TwoSlopeNorm

# sys.path.insert(0, '/Users/christinechesebrough/Documents/EPIPE-movie_nwb/Python')
sys.path.insert(0,'/media/christine/Samsung/EPIPE-movie_nwb/Python')
from epipe import read_ielvis

import os.path as op
from nibabel.freesurfer.io import read_annot


from epipe.utilities.atlases import ATLASES
from epipe.utilities.surfs import find_nearest_vertex



#%%


OMNI_VIEWS_ONE_HEM = np.array([
    ["lateral", "rostral", "dorsal"],
    ["medial", "caudal", "ventral"]
], dtype=object)

OMNI_VIEWS_BOTH_HEM = np.array([
    ["l_lateral", "l_rostral", "l_dorsal"],
    ["l_medial", "l_caudal", "l_ventral"],
    ["r_lateral", "r_rostral", "r_dorsal"],
    ["r_medial", "r_caudal", "r_ventral"]
], dtype=object)

def plot_brain_surf(
        elec_names=None,
        coords=None,
        surf='pial',
        title=None,
        subject='fsaverage',
        subjects_dir=None,
        elec_hem=None,
        snap_to_surf=False,
        views='omni',
        coord_type='LEPTO',
        max_dist=4,
        cortex='classic',
        alpha=1,
        bg_color='white',
        elec_colors='red',
        elec_size=0.5,
        cmap='coolwarm',
        cbar=False,
        cbar_title=None,
        cbar_minmax=None,
        parc=None,
        parcs_to_show=None,
        parc_colors=None,
        parc_alpha=1,
        parc_borders=False,
        vert_colors=None,
        vert_colors_alpha=None,
        clear_overlay=True,
        surface_vols=None,
        surface_vols_seg='aparc+aseg',
        surface_vol_colors=None,
        surface_vol_alpha=1,
        surface_vol_legend=False,
        return_3d=False,
        annotations=False,
        subplot_kwgs=None,
        scene_kwgs=None
):
    """Function for plotting brain surfaces with electrodes

    Parameters
    ----------
    elec_names : list | numpy.ndarray | None
        Names of electrodes. If none will load in iELViS data for the subject
    coords : numpy.ndarray, shape (n, 3) | None
        Coordinates of each electrode. If none will load in iELViS data for the subject
    surf : str
        The type of surface to show. Options are: "pial", "inflated", "flat"
    title : str
        Title of the image that will be produced
    subject : str
        Freesurfer subject ID
    subjects_dir : str | None
        Freesurfer subject directory. If not specified, will take from mne.get_config()['SUBJECTS_DIR']
    elec_hem : str | np.ndarray | list
        Specifies the hemisphere that each electrode belongs to. Each element must be a string "l" or "r"
    snap_to_surf : bool
        Whether to move electrodes to the closest vertex of the brain surface. Automatically True if surf is "pial" or "flat"
    views : str | list | numpy.ndarray
        The views the figure should show. Valid string arguements include "omni", "[l/r]omni", or "[l/r]_<view>" where <view>
        is any valid type of view including: lateral, medial, rostral, caudal, dorsal, ventral, frontal, parietal, axial, coronal.
        Can also be a 1 or 2 dimensional numpy array such as numpy.array([ ["l_lateral", "l_medial"], ["r_lateral", "r_medial"] ])
    coord_type : str
        Type of coordinates to use from iELViS
    max_dist : float | int
        When snap_to_surf=True, filter for only electrodes that are a maximum distance from the pial surface. Default=4
    cortex : str
        Style in which to display brain surfaces. Options are: classic, high_contrast, low_contrast, bone
    alpha : float
        Alpha level (opacity) of the brain surface
    bg_color : color
        background color of the 3D scene. Default="white"
    elec_colors : str | numpy.ndarray | list
        Colors of the electrodes. Can be a string or list containing elements for colors that matplotlib accepts.
        Can also be a numpy array of scalar values to color electrodes by some value
    elec_size : float | list | numpy.ndarray
        Sizes of the electrodes. Values are relative to 1cm (so 0.1 is 1mm)
    cmap : str
        Colormap to use if elec_colors are scalar values. Any matplotlib colormap. Default="coolwarm"
    cbar : bool
        Whether to include a colorbar in the figure
    cbar_title : str
        The title of the colorbar
    cbar_minmax : None | tuple | list
        A two element list or tuple of the minimum and maximum values of the colorbar. Default is to use the min and max of the scalar values
    parc : str
        The parcellation to show. Acceptable values include: dk, d, y7, y17, hcp
    parcs_to_show : list | tuple | str | None
        The parcellations of the atlas to show. If None, all parcellations are shown
    parc_colors : list | tuple | str | None
        The colors of each parcellation to show as specfied in parcs_to_show
    parc_alpha : float | None
        Alpha value (opacity) of the parcellation. Value from 0 to 1
    parc_borders : bool
        Whether to only show the borders of the parcellation or the full thing. Default is False
    vert_colors : NOT YET
    vert_colors_alpha : NOT YET
    clear_overlay : bool
        Whether prior overlays should be cleared when parcellation is added (prior overlay is typically gyri/sulci coloring).
        Default is True
    surface_vols : list | str | None
        Segmentations to render as a surface in the scene. Best used for subcortical surfaces
    surface_vols_seg : str | None
        The segmentation file to take the segmentations from. Can use shorthands "dk", and "d".
        Can also specify a file containing segmentations (such as hippocampal subfields)
    surface_vol_colors : str | list | None
        Color of each surface volume to be rendered. Can be anything matplotlib accepts
    surface_vol_alpha : float | None
        The alpha value (opacity) of the surface volumes to be rendered
    surface_vol_legend : bool
        Whether to display a legend showing the color of the surface volumes. Default is False
    return_3d : bool
        Return the 3D scene instead of a matplotlib figure. This is an interactive scene. Default is False
    annotations : bool
        For matplotlib subplots add title to each subplot that tells what each image represents (ex: l medial)
    subplot_kwgs : NOT YET
    scene_kwgs : NOT YET

    Returns
    -------
        matplotlib.figure.Figure | Figure3D
            Either a matplotlib Figure object or a 3D scene object
    """

    ###################################
    # Validate and confirm basic input
    ###################################
    if subjects_dir == None:
        subjects_dir = mne.get_config()['SUBJECTS_DIR']

    # VIEWS
    # Hemisphere(s) to use
    hems = []
    hem_views = {}

    if views == 'omni':
        plot_arrangement = OMNI_VIEWS_BOTH_HEM
        hems = ['l','r']
        hem_views['r'] = hem_views['l'] = OMNI_VIEWS_ONE_HEM.flatten().tolist()
    elif views == 'romni' or views == 'lomni':
        h = views[0]
        hems = [h]
        hem_views[h] = OMNI_VIEWS_ONE_HEM.flatten().tolist()
        plot_arrangement = OMNI_VIEWS_ONE_HEM
        for iy, ix in np.ndindex(plot_arrangement.shape):
            plot_arrangement[iy,ix] = h + '_'  + plot_arrangement[iy,ix]
    elif type(views) == str:
        plot_arrangement = np.array([views], dtype=object)
        h, hv = views.split('_')
        hem_views[h] = [hv]
        hems = [h]
    elif type(views) == list or type(views) == np.ndarray:
        views = np.array(views, dtype=object)
        plot_arrangement = views
        for v in plot_arrangement.flatten():
            h, hv = v.split('_')
            if h not in hems:
                hems.append(h)
                hem_views[h] = []
            hem_views[h].append(hv)

    if surf == 'inflated' or surf == 'flat':
        snap_to_surf = True

    ###################################
    # Electrode information
    ###################################

    # If coordinates not specified then plot subject using iELViS data
    if coords is None:
        elecs_df = read_ielvis(subject, subjects_dir)
        coords = np.array(elecs_df[coord_type].to_list())
        elec_names = elecs_df['label'].to_list()
        elec_hem = [h.lower() for h in elecs_df['hem'].to_list()]

    # Number of electrodes
    n_elecs = len(elec_names)

    # Electrodes to use
    # if type(elec_hem) == str:
    #     elec_hem = np.array([elec_hem.lower() for ii in range(n_elecs)])
    # elif type(elec_hem) == list:
    #     elec_hem = np.array(elec_hem)
    # elif elec_hem == None:
    #     elec_hem = np.array(['l' for ii in range(n_elecs)])
        
    if isinstance(elec_hem, str):
        elec_hem = np.array([elec_hem.lower() for ii in range(n_elecs)])
    elif isinstance(elec_hem, list):
        elec_hem = np.array(elec_hem)
    elif elec_hem is None:
        elec_hem = np.array(['l' for ii in range(n_elecs)])

    # Get the colors of the electrodes
    if type(elec_colors) == str or type(elec_colors) == tuple:
        elec_colors = [elec_colors for ii in range(n_elecs)]
    elif type(elec_colors) == np.ndarray:

        cmap_object = plt.get_cmap(cmap)


        # Try 1
        # elec_vals = elec_colors
        # if cbar_minmax is not None:
        #     cbar_min = cbar_minmax[0]
        #     cbar_max = cbar_minmax[0]
        # else:
        #     cbar_min = elec_vals.min()
        #     cbar_max = elec_vals.max()
        #
        # norm = mcolors.Normalize(vmin=cbar_min, vmax=cbar_max)
        # sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        # sm.set_array(elec_vals)
        # elec_colors = sm.cmap(elec_vals)
        
        elec_vals = elec_colors
        
        if cbar_minmax is not None:
            cbar_min = cbar_minmax[0]
            cbar_max = cbar_minmax[1]
        else:
            max_abs = np.nanmax(np.abs(elec_vals))
            cbar_min = -max_abs
            cbar_max = max_abs
        
        norm = mcolors.TwoSlopeNorm(vmin=cbar_min, vcenter=0, vmax=cbar_max)
        sm = plt.cm.ScalarMappable(cmap=cmap_object, norm=norm)
        sm.set_array(elec_vals)
        elec_colors = cmap_object(norm(elec_vals))

        # Try 2
        # elec_vals = elec_colors
        # elec_vals_norm = (elec_vals - elec_vals.min()) / (elec_vals.max() - elec_vals.min())
        # elec_colors = cmap_object(elec_vals_norm)
        # if cbar_minmax is not None:
        #     cbar_min = cbar_minmax[0]
        #     cbar_max = cbar_minmax[1]
        # else:
        #     cbar_min = elec_vals.min()
        #     cbar_max = elec_vals.max()
        # norm = mcolors.Normalize(vmin=cbar_min, vmax=cbar_max)
        # sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        # sm.set_array(elec_vals)

    elec_colors = np.array(elec_colors)

    # Set electrode sizes
    if type(elec_size) != list and type(elec_size) != np.ndarray:
       elec_size = [elec_size for ii in range(n_elecs)]
    elec_size = np.array(elec_size)

    # Names
    elec_names = np.array(elec_names)

    ###################################
    # Set file paths
    ###################################
    sub_dir = op.join(subjects_dir, subject)
    label_dir = op.join(sub_dir, 'label')

    ###################################
    # Plot the 3D brain
    ###################################
    if return_3d:
        mne_fig = mne.viz.create_3d_figure((1000,1000), show=True, bgcolor=bg_color)

    screenshots = {}

    for h in hems:

        if not return_3d:
            mne_fig = mne.viz.create_3d_figure((1000, 1000), show=True, bgcolor=bg_color)

        # The brain surf
        brain = mne.viz.Brain(
            subject,
            hemi=h + 'h',  # 'lh', 'rh', 'both'
            subjects_dir=subjects_dir,
            surf=surf,  # 'pial', 'white', 'inflated', 'flat'
            figure=mne_fig,
            cortex=cortex,
            background=bg_color,
            alpha=alpha)

        ###################################
        # Add the electrodes
        ###################################

        # Index which electrodes belong to this hemisphere
        this_hem_elecs_idx = elec_hem == h
        n_elecs_this_hem = this_hem_elecs_idx.sum()
        hem_coords = coords[this_hem_elecs_idx,:]
        hem_colors = elec_colors[this_hem_elecs_idx]
        hem_size = elec_size[this_hem_elecs_idx]
        hem_names = elec_names[this_hem_elecs_idx]
        hem_dist = np.zeros(n_elecs_this_hem)

        if len(hem_coords) == 0:
            continue

        # Snap to surface if need be
        if snap_to_surf:
            vertex_df = find_nearest_vertex(subject, subjects_dir=subjects_dir, coords=hem_coords, hem=h, labels=hem_names)
            hem_coords = vertex_df['closest_vert'].to_numpy()
            hem_dist = vertex_df['distance'].to_numpy()

        # Add electrodes to the scene one-by-one
        # This is slower than adding them all at once but allows customization of appearance of each electrode
        for elec_ii in range(n_elecs_this_hem):
            if snap_to_surf and max_dist < hem_dist[elec_ii]:
                continue
            else:
                brain.add_foci(
                    coords=np.array(hem_coords[elec_ii]),
                    color=hem_colors[elec_ii],
                    scale_factor=hem_size[elec_ii],
                    coords_as_verts=snap_to_surf
                )

        ###################################
        # Add overlays/parcellations
        ###################################
        if parc is not None:
            if op.isfile(parc):
                p_labels, p_ctab, p_names = read_annot(parc, orig_ids=True)
                p_names = [p.decode('UTF-8') for p in p_names]
            elif parc in ATLASES.keys():
                parc_file = op.join(label_dir, h + 'h.' + ATLASES[parc] + '.annot')
                p_labels, p_ctab, p_names = read_annot(parc_file, orig_ids=True)
                p_names = [p.decode('UTF-8') for p in p_names]

            # If select parcels are to be shown
            if type(parcs_to_show) == str:
                parcs_to_show = [parcs_to_show]
            if type(parc_colors) == str:
                parc_colors = [parc_colors]


            if parcs_to_show is None:
                brain.add_annotation(annot=(p_labels, p_ctab), hemi=h + 'h', borders=parc_borders, alpha=parc_alpha,
                                     remove_existing=clear_overlay)
            else:
                for ii in range(len(parcs_to_show)):
                    this_parc_idx = p_names.index(parcs_to_show[ii])
                    this_parc_id = p_ctab[this_parc_idx,4]
                    l = mne.Label(vertices=np.where(p_labels==this_parc_id)[0], color=parc_colors[ii], hemi=h+'h')
                    brain.add_label(l,color=parc_colors[ii],hemi=h+'h')

        ###################################
        # Add Surface Volumes
        ###################################
        if surface_vols is not None:

            if type(surface_vols) == str:
                surface_vols = [surface_vols]
            if type(surface_vol_colors) == str:
                surface_vol_colors == [surface_vol_colors]

            if surface_vols_seg in ATLASES.keys():
                surface_vols_seg = ATLASES[surface_vols_seg] + '+aseg'
            elif not op.isfile(surface_vols_seg):
                raise ValueError("%s is not a file or a known atlas" % surface_vols_seg)

            brain.add_volume_labels(
                labels=surface_vols,
                colors=surface_vol_colors,
                aseg=surface_vols_seg,
                legend=surface_vol_legend,
                alpha=surface_vol_alpha
            )

        ###########################################
        # Take screenshots in the views specified
        ###########################################
        if return_3d:
            continue
        else:
            for v in hem_views[h]:
                v_key = h + '_' + v
                brain.show_view(view=v, hemi=h + 'h')
                screenshots[f'{v_key}'] = brain.screenshot()
            mne.viz.close_3d_figure(mne_fig)

    ###########################################
    # Return 3D scene or matplotlib figure
    ###########################################
    if return_3d:
        mne_fig.plotter.add_menu_bar()
        return mne_fig
    else:

        if plot_arrangement.ndim == 1:
            plot_arrangement = plot_arrangement.reshape(1, plot_arrangement.size)

        mplot_fig, axes = plt.subplots(*plot_arrangement.shape, figsize=(5, 8))
        mplot_fig.suptitle(title, fontsize=12)

        if plot_arrangement.size > 1:

            axes = axes.reshape(plot_arrangement.shape)
            for iy, ix in np.ndindex(plot_arrangement.shape):
                v_key = plot_arrangement[iy, ix]
                axes[iy,ix].imshow(screenshots[v_key])
                axes[iy,ix].set_axis_off()
                if annotations:
                    axes[iy, ix].set_title( v_key.replace('_', ' ') )
        else:
            axes.imshow( screenshots[list(screenshots.keys())[0]] )
            axes.set_axis_off()

        mplot_fig.tight_layout()

        if cbar:
            mplot_fig.colorbar(mappable=sm, ax=axes, location='right', label=cbar_title)

        return mplot_fig


# import matplotlib.colors as mcolors
# np.array(mcolors.to_rgba("r")).astype(int)*255
def _color_to_rgba(color):
    from PIL import ImageColor
    return ImageColor.getcolor(color, "RGB")


#%%
subjects_dir = '/media/christine/Data/anatomy'

pat = 'NS219'
atlas = 'All'
model = "LDA"
label_scheme = 'extremes'


if pat == 'LH019':
    views = [["l_lateral"], ["l_medial"],["l_ventral"]]
else:
    views=[["l_lateral","r_lateral"], ["l_medial",'r_medial'],["l_ventral","r_ventral"]]

results_file = f"/media/christine/Samsung/attn_states_decoding/outputs/{model}/{pat}/{pat}_lda_contact_attention_results_{label_scheme}_{atlas}_All.csv"
#results_file = f"/media/christine/Samsung/attn_states_decoding/outputs/LDA/{pat}/{pat}_lda_contact_attention_results_thirds_{atlas}_All.csv"

img_dir = f"/media/christine/Samsung/attn_states_decoding/outputs/{pat}"

results_df = pd.read_csv(results_file)

freq_bands = ['HFA','alpha','theta','beta','gamma']

#freq_bands = ['HFA']


for freq in freq_bands:
    
    plot_df = results_df[results_df["frequency"] == freq].copy()
    
    coords = plot_df[["fsaverage_coords_1", "fsaverage_coords_2", "fsaverage_coords_3"]].to_numpy()
    elec_names = plot_df["contact"].to_numpy()
    coefficients = plot_df["mean_coefficient"].to_numpy()
    
    elec_hem = np.where(plot_df["contact"].str.upper().str.startswith("L"), "l", "r")
    
    fig = plot_brain_surf(
        elec_names=elec_names,
        coords=coords,
        surf="inflated",
        title=f"{pat} {model} coefficients - {freq}",
        subject="fsaverage",
        elec_size=1,
        subjects_dir=subjects_dir,
        elec_hem=elec_hem,
        snap_to_surf=True,
        views=views,
        elec_colors=coefficients,
        cmap="coolwarm",
        cbar=True,
        cbar_title="LDA coefficient",
        parc_alpha=1,
        return_3d = False        # translucent parcels so electrodes remain visible
    
    )
    
    out_path = os.path.join(
        img_dir,
        f"{freq}_{atlas}_contact_coefficients_brainplot.png")
    
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    print("Saved:", out_path)

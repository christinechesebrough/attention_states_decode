from nilearn import plotting
import os
import os.path as op
from epipe import read_ielvis
import matplotlib.pyplot as plt
import nibabel as nib

def plot_ortho(
        volume="brainmask",
        subject=None,
        subjects_dir=None,
        coord=None,
        elec_name=None,
        elec_size=40,
        annotate=True,
        draw_cross=False,
        title=None,
        threshold=10,
        coord_type='LEPTO',
        display_mode='ortho',
        show=True,
        fig_kwgs={'figsize': (10,5)}
):
    if subjects_dir is None:
        import mne
        subjects_dir = mne.get_config()['SUBJECTS_DIR']

    sub_dir = op.join(subjects_dir, subject)

    if isinstance(volume, nib.freesurfer.mghformat.MGHImage) or op.isfile(volume):
        pass
    else:
        volume = op.join(sub_dir, 'elec_recon', volume + '.nii.gz')

    if type(elec_name) == str:
        elecs = read_ielvis(sub_dir)
        elecs = elecs.set_index('label')
        this_elec = elecs.loc[elec_name]
        coord = this_elec[coord_type]
        title = elec_name

    # Create the figure
    fig = plt.figure(**fig_kwgs)
    brain_fig = plotting.plot_anat(
        volume,
        title=title,
        display_mode=display_mode,
        cut_coords=coord,
        draw_cross=draw_cross,
        annotate=annotate,
        figure=fig,
        threshold=threshold)
    brain_fig.add_markers(marker_coords=[coord], marker_size=elec_size)
    #plt.show()

    return fig
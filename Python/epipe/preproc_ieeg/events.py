import numpy as np
import matplotlib.pyplot as plt
import neurokit2 as nk

# TODO: ana2dig also can return offsets of pulses for stimulations


def ana2dig(ana,fs=None,thr="auto",min_diff=10,return_time=False,plot=False):
    """Function for converting an analog signal into a digital one and getting onsets

    Args:
        ana: The analog array to digitize
        fs: Sampling rate of the signal (optional)
        thr: Threshold of signal to help with binarization
        min_diff: Minimum number of samples the digital onsets must be separated by. If "fs" is not None then it is minimum amount of time
        return_time: Boolean. Whether to return the time at which the the digital onset occurred
        plot: Boolean. If the result of this function should be plotted for visual inspection


    Returns:
        ana_bin: The analog signal binarized
        dig_times: Only returned if "return_time=True". Returns the onset sample of each discrete pulse. If fs does not equal None then timestamps are returned

    """

    ana = ana.flatten()

    if fs != None:
        tvec = np.arange(0,len(ana),1) / fs
        max_diff_samples = int( np.ceil(min_diff*fs) )
    else:
        tvec = np.arange(0,len(ana),1)
        max_diff_samples = min_diff

    # Binarize analog signal
    ana_bin = nk.signal_binarize(ana, threshold=thr)
    ana_high = np.where(ana_bin)[0]

    # Now check how far apart the points are.
    # If they're too close, they're part of a previous pulse
    ana_high_diff = np.diff(ana_high)
    ana_high_diff_issmall = ana_high_diff <= max_diff_samples

    # Remove points that aren't true digital points
    not_dig = np.where(ana_high_diff_issmall)[0] + 1
    dig_onset_points = np.delete(ana_high, not_dig)

    # Get times of digital onsets
    is_dig_onset = np.zeros((len(ana)))
    is_dig_onset[dig_onset_points] = 1
    dig_times = tvec[dig_onset_points]

    # Plot to check
    if plot:
        fig, ax = plt.subplots()
        ax.plot(tvec, ana, 'r', label='Analog')
        ymin, ymax = ax.get_ylim()
        ax.vlines(dig_times, ymin, ymax, 'b')
        plt.show()

    if return_time:
        return ana_bin, dig_times
    else:
        return ana_bin


def create_toi_slice(tvec,t1,t2,tstep):
    """Create a slice object that can be used to decimate MNE-Python objects"""

    start_idx = np.abs(tvec - t1).argmin()
    step1_idx = np.abs(tvec - (t1 + tstep) ).argmin()
    stride = step1_idx - start_idx
    toi = slice(
        np.abs(tvec - t1).argmin(),
        np.abs(tvec - t2).argmin() + 1,
        stride)
    return toi


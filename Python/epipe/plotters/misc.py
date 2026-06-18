import numpy as np
import matplotlib.pyplot as plt

def plot_stem_annotation(data, labels, threshold=None, title=None):
    """Create a stem plot that annotates points that surpass the threshold

    Parameters
    ----------
    data
    labels
    threshold
    title

    Returns
    -------

    """

    # Create the stem plot
    fig = plt.figure(figsize=(10, 6))
    markerline, stemlines, baseline = plt.stem(data, markerfmt='o', linefmt='gray', basefmt='gray')
    plt.setp(markerline, color='blue', markersize=8)
    plt.setp(stemlines, color='gray')
    plt.setp(baseline, color='gray')

    _, ymax = plt.ylim()
    plt.ylim(0, ymax * 1.2)
    txt_offset = plt.ylim()[1] * 0.1

    # Highlight data points above the threshold
    above_threshold_indices = np.where(data > threshold)[0]
    plt.plot(above_threshold_indices, data[above_threshold_indices], 'ro')

    for index in above_threshold_indices:
        plt.annotate(labels[index], (index, data[index] + txt_offset), textcoords="offset points", xytext=(0, 0),
                     ha='center', fontsize=8, color='red')
        plt.plot([index, index], [data[index], data[index] + txt_offset], 'k', linewidth=0.2)

    # Add title and labels
    plt.title(title)
    plt.show(block=False)
    plt.ioff()
    plt.xticks([])

    return fig


"""
Movies ET Attention Main - Source Utilities

This package contains reusable functions for EEG preprocessing and eye tracking analysis.
"""

from .eeg_preproc_helpers import (
    plot_psd_batched,
    ProcessingLogger,
    save_bad_channels,
    load_bad_channels,
    create_file_paths
)

# from .eye_utils import (
#     # Add your eye tracking functions here
#     # process_gaze_data,
#     # compute_eye_measures,
#     # etc.
# )

__version__ = "0.1.0"
__author__ = "Christine Chesebrough" 
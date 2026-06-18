# Movies ET Attention Main

A comprehensive package for processing movie-watching EEG and eye tracking data.

## Structure

```
movies_ET_attn_main/
├── scripts/              # Main analysis scripts
│   ├── movie_ieeg_preprocess.py  # EEG preprocessing pipeline
│   ├── compute_eye_measures.py   # Eye tracking analysis
│   ├── compute_vergence.py       # Vergence calculation
│   ├── gaze_stats_isc.py         # Gaze ISC analysis
│   └── preprocess_eye_measures.py # Eye measures preprocessing
├── src/                  # Reusable functions
│   ├── eeg_preproc_helpers.py    # EEG preprocessing utilities
│   └── eye_helpers.py            # Eye tracking utilities
├── config/               # Configuration files
├── tests/                # Test files and data
└── README.md             # This file
```

## Usage

### EEG Preprocessing
```python
# Import utilities
from src.eeg_preproc_helpers import plot_psd_batched, ProcessingLogger

# Use in your scripts
figures = plot_psd_batched(raw_data, batch_size=32)
```

### Eye Tracking Analysis
```python
# Import eye tracking utilities
from src.eye_helpers import calculate_vergence_measures, detect_blinks

# Calculate vergence measures
vergence_data = calculate_vergence_measures(nwb_data, screen_pix, screen_cm)

# Detect blinks
blink_events = detect_blinks(val_left, val_right, t_nwb, fs_eye)
```

### Vergence Calculation
```python
# Run vergence calculation script
python scripts/vergence_calc.py

# The script will:
# - Calculate horizontal and vertical gaze disparity
# - Compute visual focus displacement (Huang et al. 2019)
# - Detect and save blink events
# - Generate plots and save results to CSV
# - Output files compatible with compute_eye_measures.py
```

## Dependencies

- numpy
- pandas
- matplotlib
- mne
- scipy
- pynwb

## Installation

1. Clone or download this repository
2. Add the `src/` directory to your Python path:
   ```python
   import sys
   sys.path.append('/path/to/movies_ET_attn_main/src')
   ```

## Contributing

Please follow the existing code structure and add documentation for new functions. 
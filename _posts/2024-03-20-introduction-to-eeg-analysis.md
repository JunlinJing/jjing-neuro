---
layout: post
title: "Introduction to EEG Analysis with Python"
date: 2024-03-20
description: "A comprehensive guide to getting started with EEG data analysis using Python and MNE library"
tag: 
  - EEG
  - Python
  - Neuroscience
---

# Introduction to EEG Analysis with Python

Electroencephalography (EEG) is a powerful tool for studying brain activity. In this tutorial, we'll explore how to analyze EEG data using Python and the popular MNE library.

## Setting Up Your Environment

First, let's set up our Python environment with the necessary packages:

```python
import mne
import numpy as np
import matplotlib.pyplot as plt

# For data visualization
%matplotlib inline
```

## Loading and Preprocessing EEG Data

Here's a basic example of loading and preprocessing EEG data:

```python
# Load example data
sample_data_folder = mne.datasets.sample.data_path()
raw_fname = sample_data_folder + '/MEG/sample/sample_audvis_raw.fif'
raw = mne.io.read_raw_fif(raw_fname, preload=True)

# Basic preprocessing
raw.filter(1, 40)  # Band-pass filter from 1-40 Hz
raw.notch_filter(60)  # Remove power line noise
```

## Analyzing EEG Data

Let's look at some basic analysis techniques:

```python
# Create epochs
events = mne.find_events(raw)
epochs = mne.Epochs(raw, events, event_id=1, tmin=-0.2, tmax=0.5)

# Calculate and plot evoked response
evoked = epochs.average()
evoked.plot()
```

## Time-Frequency Analysis

Here's how to perform time-frequency analysis:

```python
frequencies = np.arange(1, 40, 1)
power = mne.time_frequency.tfr_morlet(epochs, frequencies, 
                                    n_cycles=2, return_itc=False)
power.plot([0])
```

## EEG Band Analysis

A crucial part of EEG analysis is examining different frequency bands:

```python
def analyze_frequency_bands(raw, picks=['Fz', 'Cz', 'Pz']):
    """Analyze standard EEG frequency bands"""
    # Define frequency bands
    bands = {
        'Delta': (0.5, 4),
        'Theta': (4, 8),
        'Alpha': (8, 13),
        'Beta': (13, 30),
        'Gamma': (30, 45)
    }
    
    # Calculate power spectral density
    psds, freqs = mne.time_frequency.psd_welch(raw, 
                                              fmin=0.5,
                                              fmax=45,
                                              picks=picks)
    
    # Calculate band power
    band_powers = {}
    for band, (fmin, fmax) in bands.items():
        freq_mask = (freqs >= fmin) & (freqs <= fmax)
        band_powers[band] = np.mean(psds[:, freq_mask], axis=1)
    
    return band_powers

# Example visualization
def plot_band_powers(band_powers, channel_names):
    """Plot power in different frequency bands"""
    bands = list(band_powers.keys())
    channels = len(channel_names)
    
    plt.figure(figsize=(12, 6))
    x = np.arange(len(bands))
    width = 0.8 / channels
    
    for i, channel in enumerate(channel_names):
        powers = [band_powers[band][i] for band in bands]
        plt.bar(x + i * width, powers, width, label=channel)
    
    plt.xlabel('Frequency Bands')
    plt.ylabel('Power (µV²/Hz)')
    plt.title('EEG Band Powers')
    plt.xticks(x + width * (channels-1)/2, bands)
    plt.legend()
    plt.tight_layout()
```

## Data Quality Assessment

Before detailed analysis, it's important to assess data quality:

```python
def check_data_quality(raw):
    """Basic data quality checks"""
    # Check for flat signals
    flat_channels = []
    for ch_idx in range(len(raw.ch_names)):
        if np.std(raw._data[ch_idx]) < 1e-6:
            flat_channels.append(raw.ch_names[ch_idx])
    
    # Check for noisy channels
    noisy_channels = []
    zscore_thresh = 4.0
    for ch_idx in range(len(raw.ch_names)):
        z_scores = np.abs(stats.zscore(raw._data[ch_idx]))
        if np.any(z_scores > zscore_thresh):
            noisy_channels.append(raw.ch_names[ch_idx])
    
    return {
        'flat_channels': flat_channels,
        'noisy_channels': noisy_channels,
        'total_channels': len(raw.ch_names),
        'duration': raw.times[-1],
        'sampling_rate': raw.info['sfreq']
    }
```

## Conclusion

This introduction covers the basics of EEG analysis with Python. In future posts, we'll explore more advanced topics like source localization and connectivity analysis.

## References

1. Gramfort, A., et al. (2013). MEG and EEG data analysis with MNE-Python. Frontiers in Neuroscience.
2. Cohen, M. X. (2014). Analyzing Neural Time Series Data: Theory and Practice. 
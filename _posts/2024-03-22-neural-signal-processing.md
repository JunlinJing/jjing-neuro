---
layout: post
title: "Advanced Neural Signal Processing Techniques"
date: 2024-03-22
description: "A deep dive into modern neural signal processing methods with practical Python implementations"
tag: 
  - Signal Processing
  - Neuroscience
  - Python
category: blog
---

# Advanced Neural Signal Processing Techniques

Neural signal processing is fundamental to understanding brain function and developing neural interfaces. This post explores advanced techniques for processing and analyzing neural signals.

## Wavelet Analysis

Wavelets are particularly useful for analyzing time-frequency characteristics of neural signals:

```python
import pywt
import numpy as np
import matplotlib.pyplot as plt

def wavelet_analysis(signal, sampling_rate):
    """Perform wavelet analysis on neural signals"""
    # Choose wavelet type and levels
    wavelet = 'db4'
    levels = 5
    
    # Perform wavelet decomposition
    coeffs = pywt.wavedec(signal, wavelet, level=levels)
    
    # Reconstruct signals at each level
    reconstructed = []
    for i in range(levels):
        coeff_list = [np.zeros_like(c) for c in coeffs]
        coeff_list[i] = coeffs[i]
        reconstructed.append(pywt.waverec(coeff_list, wavelet))
    
    return reconstructed

# Example usage
def plot_wavelet_decomposition(signal, sampling_rate):
    reconstructed = wavelet_analysis(signal, sampling_rate)
    time = np.arange(len(signal)) / sampling_rate
    
    plt.figure(figsize=(12, 8))
    for i, rec in enumerate(reconstructed):
        plt.subplot(len(reconstructed), 1, i+1)
        plt.plot(time, rec)
        plt.title(f'Level {i+1} Decomposition')
    plt.tight_layout()
```

## Advanced Filtering Techniques

Implementation of advanced filtering methods for neural signals:

```python
from scipy import signal

def design_filters():
    """Design various types of filters for neural signal processing"""
    
    def notch_filter(data, freq, q, fs):
        """Notch filter for removing power line noise"""
        b, a = signal.iirnotch(freq, q, fs)
        return signal.filtfilt(b, a, data)
    
    def bandpass_filter(data, low_freq, high_freq, fs, order=4):
        """Bandpass filter for isolating frequency bands of interest"""
        nyq = fs * 0.5
        low = low_freq / nyq
        high = high_freq / nyq
        b, a = signal.butter(order, [low, high], btype='band')
        return signal.filtfilt(b, a, data)
    
    def savitzky_golay_filter(data, window_length, polyorder):
        """Savitzky-Golay filter for smoothing while preserving high moments"""
        return signal.savgol_filter(data, window_length, polyorder)
    
    return notch_filter, bandpass_filter, savitzky_golay_filter
```

## Feature Extraction

Common feature extraction methods for neural signals:

```python
def extract_features(signal, fs):
    """Extract common features from neural signals"""
    features = {}
    
    # Time domain features
    features['mean'] = np.mean(signal)
    features['std'] = np.std(signal)
    features['var'] = np.var(signal)
    features['rms'] = np.sqrt(np.mean(signal**2))
    
    # Frequency domain features
    freqs, psd = signal.welch(signal, fs, nperseg=256)
    features['peak_freq'] = freqs[np.argmax(psd)]
    features['mean_freq'] = np.sum(freqs * psd) / np.sum(psd)
    
    # Entropy
    features['sample_entropy'] = compute_sample_entropy(signal)
    
    return features

def compute_sample_entropy(signal, m=2, r=0.2):
    """Compute sample entropy of the signal"""
    # Implementation of sample entropy calculation
    N = len(signal)
    r = r * np.std(signal)
    
    def count_matches(template, m):
        count = 0
        for i in range(N - m + 1):
            if np.all(np.abs(signal[i:i+m] - template) < r):
                count += 1
        return count - 1  # Subtract self-match
    
    # Count matches for m and m+1 length templates
    B = sum(count_matches(signal[i:i+m], m) for i in range(N-m+1))
    A = sum(count_matches(signal[i:i+m+1], m+1) for i in range(N-m))
    
    return -np.log(A/B)
```

## Artifact Removal

Implementation of ICA-based artifact removal:

```python
from sklearn.decomposition import FastICA

def remove_artifacts_ica(eeg_data, n_components=None):
    """Remove artifacts using Independent Component Analysis"""
    # Reshape data if needed
    if eeg_data.ndim == 3:  # trials x channels x time
        trials, channels, samples = eeg_data.shape
        eeg_data = eeg_data.reshape(trials * samples, channels)
    
    # Apply ICA
    ica = FastICA(n_components=n_components, random_state=42)
    components = ica.fit_transform(eeg_data)
    
    # Here you would typically identify artifact components
    # This is often done through visual inspection or automated methods
    
    # Reconstruct signal without artifact components
    cleaned_data = ica.inverse_transform(components)
    
    return cleaned_data
```

## Advanced Connectivity Analysis

Modern neural signal analysis often involves studying connectivity between different brain regions:

```python
def compute_connectivity(signals, method='plv', fs=256):
    """Compute various connectivity measures between signals"""
    if method == 'plv':
        # Phase Locking Value
        def compute_plv(x, y):
            analytic_x = signal.hilbert(x)
            analytic_y = signal.hilbert(y)
            phase_diff = np.angle(analytic_x) - np.angle(analytic_y)
            return np.abs(np.mean(np.exp(1j * phase_diff)))
        
        n_channels = signals.shape[0]
        connectivity = np.zeros((n_channels, n_channels))
        
        for i in range(n_channels):
            for j in range(i+1, n_channels):
                plv = compute_plv(signals[i], signals[j])
                connectivity[i,j] = plv
                connectivity[j,i] = plv
                
        return connectivity
    
    elif method == 'coherence':
        # Magnitude Squared Coherence
        n_channels = signals.shape[0]
        connectivity = np.zeros((n_channels, n_channels))
        
        for i in range(n_channels):
            for j in range(i+1, n_channels):
                f, coh = signal.coherence(signals[i], signals[j], fs=fs)
                connectivity[i,j] = np.mean(coh)
                connectivity[j,i] = connectivity[i,j]
                
        return connectivity

def plot_connectivity(connectivity, channel_names):
    """Plot connectivity matrix as a heatmap"""
    plt.figure(figsize=(10, 8))
    plt.imshow(connectivity, cmap='viridis', aspect='equal')
    plt.colorbar(label='Connectivity Strength')
    plt.xticks(range(len(channel_names)), channel_names, rotation=45)
    plt.yticks(range(len(channel_names)), channel_names)
    plt.title('Brain Connectivity Matrix')
    plt.tight_layout()
```

## Advanced Time-Series Analysis

Implementation of advanced time series analysis methods:

```python
def compute_multiscale_entropy(signal, scales=20, m=2, r=0.15):
    """Compute Multiscale Entropy"""
    def coarse_grain(data, scale):
        """Coarse-graining for multiscale entropy"""
        n = len(data)
        coarse = np.zeros(n // scale)
        for i in range(0, n - scale + 1, scale):
            coarse[i // scale] = np.mean(data[i:i+scale])
        return coarse
    
    # Normalize signal
    signal = (signal - np.mean(signal)) / np.std(signal)
    mse = np.zeros(scales)
    
    for scale in range(1, scales + 1):
        coarse_signal = coarse_grain(signal, scale)
        mse[scale-1] = compute_sample_entropy(coarse_signal, m, r)
    
    return mse

def detrended_fluctuation_analysis(signal, scales=None):
    """Perform Detrended Fluctuation Analysis"""
    if scales is None:
        scales = np.logspace(1, np.log10(len(signal)//4), 20).astype(int)
    
    # Calculate profile
    profile = np.cumsum(signal - np.mean(signal))
    
    fluctuations = np.zeros(len(scales))
    
    for i, scale in enumerate(scales):
        # Split signal into windows
        n_windows = len(profile) // scale
        windows = np.array_split(profile[:n_windows*scale], n_windows)
        
        # Calculate local trend and fluctuation
        x = np.arange(scale)
        fluct = np.zeros(n_windows)
        
        for j, window in enumerate(windows):
            coef = np.polyfit(x, window, 1)
            trend = np.polyval(coef, x)
            fluct[j] = np.sqrt(np.mean((window - trend)**2))
            
        fluctuations[i] = np.mean(fluct)
    
    # Calculate scaling exponent
    coef = np.polyfit(np.log(scales), np.log(fluctuations), 1)
    return scales, fluctuations, coef[0]
```

## Non-linear Dynamics Analysis

Tools for analyzing non-linear dynamics in neural signals:

```python
def recurrence_plot(signal, dimension=3, delay=1, threshold=0.1):
    """Generate a recurrence plot from time series data"""
    # Time delay embedding
    N = len(signal) - (dimension - 1) * delay
    phase_space = np.zeros((N, dimension))
    
    for i in range(dimension):
        phase_space[:, i] = signal[i*delay:i*delay + N]
    
    # Calculate distances
    distances = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            distances[i,j] = np.sqrt(np.sum((phase_space[i] - phase_space[j])**2))
    
    # Create recurrence matrix
    recurrence = distances < threshold
    
    return recurrence

def lyapunov_exponent(signal, dimension=3, delay=1, dt=1.0):
    """Estimate the largest Lyapunov exponent"""
    # Phase space reconstruction
    N = len(signal) - (dimension - 1) * delay
    phase_space = np.zeros((N, dimension))
    
    for i in range(dimension):
        phase_space[:, i] = signal[i*delay:i*delay + N]
    
    # Find nearest neighbors
    divergences = np.zeros(N)
    for i in range(N):
        distances = np.sqrt(np.sum((phase_space - phase_space[i])**2, axis=1))
        nearest = np.argsort(distances)[1]  # Exclude self
        
        # Track divergence
        divergences[i] = np.log(np.abs(signal[i] - signal[nearest])) / dt
    
    return np.mean(divergences)
```

## Conclusion

These advanced signal processing techniques form the foundation for modern neural signal analysis. They enable researchers to extract meaningful information from complex neural recordings and develop more sophisticated brain-computer interfaces.

## References

1. Cohen, M. X. (2014). Analyzing Neural Time Series Data: Theory and Practice.
2. Makeig, S., et al. (2004). Mining event-related brain dynamics.
3. Quiroga, R. Q., et al. (2004). Independent component analysis for neural signal processing. 
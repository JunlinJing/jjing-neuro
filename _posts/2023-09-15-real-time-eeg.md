---
layout: post
title: "Building Real-time EEG Signal Processing Pipeline"
date: 2023-09-15 16:00
image: /assets/images/blog/real-time-eeg.jpg
headerImage: true
tag:
- EEG
- signal processing
- real-time
- python
category: blog
author: jimjing
description: A detailed guide on building real-time EEG signal processing systems using Python
---

# Building Real-time EEG Signal Processing Pipeline

Real-time EEG signal processing is crucial for brain-computer interfaces and neurofeedback applications. This tutorial shows how to build an efficient real-time processing pipeline using Python.

## System Architecture

A real-time EEG processing system typically consists of:
1. Data acquisition
2. Signal preprocessing
3. Feature extraction
4. Classification/Analysis
5. Feedback generation

### Basic Structure

```python
class RealTimeEEG:
    def __init__(self, device, sampling_rate, channels):
        self.device = device
        self.fs = sampling_rate
        self.channels = channels
        self.buffer_size = int(self.fs * 2)  # 2-second buffer
        self.buffer = np.zeros((len(channels), self.buffer_size))
        
    def initialize(self):
        self.setup_filters()
        self.setup_features()
        self.setup_classifier()
```

## Data Acquisition

### Setting up LSL Stream

```python
from pylsl import StreamInlet, resolve_stream

def setup_eeg_stream():
    """Setup LSL stream for EEG data."""
    streams = resolve_stream('type', 'EEG')
    inlet = StreamInlet(streams[0])
    return inlet

def acquire_data(inlet, chunk_size):
    """Acquire data chunks from LSL stream."""
    chunk, timestamps = inlet.pull_chunk(
        max_samples=chunk_size,
        timeout=1.0
    )
    return np.array(chunk), timestamps
```

## Real-time Preprocessing

### Filtering

```python
from scipy.signal import butter, sosfilt

class RealTimeFilter:
    def __init__(self, fs, f_low, f_high, order=4):
        self.fs = fs
        self.f_low = f_low
        self.f_high = f_high
        self.order = order
        self.sos = self._design_filter()
        
    def _design_filter(self):
        nyq = self.fs / 2
        sos = butter(self.order, 
                    [self.f_low/nyq, self.f_high/nyq],
                    btype='bandpass',
                    output='sos')
        return sos
    
    def apply(self, data):
        return sosfilt(self.sos, data, axis=-1)
```

### Artifact Removal

```python
class ArtifactRemover:
    def __init__(self, threshold=100):
        self.threshold = threshold
        
    def remove_artifacts(self, data):
        """Simple threshold-based artifact removal."""
        mask = np.abs(data) > self.threshold
        data[mask] = np.nan
        # Interpolate NaN values
        data = pd.DataFrame(data).interpolate(method='linear')
        return data.values
```

## Feature Extraction

### Real-time Features

```python
class RealTimeFeatures:
    def __init__(self, fs, window_size=1.0):
        self.fs = fs
        self.window_size = window_size
        self.window_samples = int(fs * window_size)
        
    def compute_features(self, data):
        """Compute features in real-time."""
        features = {}
        
        # Time domain features
        features['rms'] = np.sqrt(np.mean(np.square(data)))
        features['var'] = np.var(data)
        
        # Frequency domain features
        freqs, psd = signal.welch(data, 
                                fs=self.fs,
                                nperseg=self.window_samples,
                                noverlap=self.window_samples//2)
        
        # Band powers
        bands = {
            'delta': (0.5, 4),
            'theta': (4, 8),
            'alpha': (8, 13),
            'beta': (13, 30)
        }
        
        for band, (fmin, fmax) in bands.items():
            idx = np.logical_and(freqs >= fmin, freqs <= fmax)
            features[f'{band}_power'] = np.mean(psd[idx])
            
        return features
```

## Real-time Processing Pipeline

### Main Processing Loop

```python
class EEGProcessor:
    def __init__(self, fs=250, channels=['Fp1', 'Fp2', 'C3', 'C4']):
        self.fs = fs
        self.channels = channels
        self.filter = RealTimeFilter(fs, 1, 40)
        self.artifact_remover = ArtifactRemover()
        self.feature_extractor = RealTimeFeatures(fs)
        
    def process_chunk(self, data):
        """Process a chunk of EEG data in real-time."""
        # Filter
        filtered = self.filter.apply(data)
        
        # Remove artifacts
        clean = self.artifact_remover.remove_artifacts(filtered)
        
        # Extract features
        features = {}
        for i, ch in enumerate(self.channels):
            ch_features = self.feature_extractor.compute_features(
                clean[i, :]
            )
            features[ch] = ch_features
            
        return features
```

## Visualization

### Real-time Plot

```python
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

class RealTimePlot:
    def __init__(self, fs, channels, buffer_size=1000):
        self.fs = fs
        self.channels = channels
        self.buffer_size = buffer_size
        self.setup_plot()
        
    def setup_plot(self):
        self.fig, self.ax = plt.subplots()
        self.lines = []
        for ch in self.channels:
            line, = self.ax.plot([], [], label=ch)
            self.lines.append(line)
        self.ax.set_ylim(-100, 100)
        self.ax.legend()
        
    def update(self, frame, data):
        for i, line in enumerate(self.lines):
            line.set_data(range(len(data[i])), data[i])
        return self.lines
```

## Complete System

Here's how to put everything together:

```python
def run_eeg_system():
    # Initialize components
    fs = 250
    channels = ['Fp1', 'Fp2', 'C3', 'C4']
    processor = EEGProcessor(fs, channels)
    inlet = setup_eeg_stream()
    plotter = RealTimePlot(fs, channels)
    
    # Main loop
    while True:
        # Get data
        data, timestamps = acquire_data(inlet, chunk_size=fs//10)
        
        # Process
        features = processor.process_chunk(data)
        
        # Update visualization
        plotter.update(None, data)
        
        # Optional: Save or stream features
        save_features(features)
        
        # Check for exit condition
        if check_exit_condition():
            break
```

## Performance Optimization

1. **Buffer Management**
   - Use circular buffers
   - Optimize memory allocation
   - Minimize copying

2. **Computational Efficiency**
   - Vectorize operations
   - Use parallel processing
   - Profile and optimize bottlenecks

3. **Latency Reduction**
   - Minimize buffer sizes
   - Optimize filter designs
   - Use efficient data structures

## Best Practices

1. **System Design**
   - Modular architecture
   - Error handling
   - Data validation

2. **Real-time Considerations**
   - Timing accuracy
   - Buffer overruns
   - Resource management

3. **Quality Control**
   - Signal quality metrics
   - Performance monitoring
   - System diagnostics

## Conclusion

Building a real-time EEG processing pipeline requires careful consideration of various factors including data acquisition, processing efficiency, and system reliability. This guide provides a foundation for developing such systems.

## References

1. Brunner, C. et al. (2015). "BCI Software Platforms"
2. Delorme, A. & Makeig, S. (2004). "EEGLAB"
3. Renard, Y. et al. (2010). "OpenViBE"

## Resources

- [LSL Documentation](https://labstreaminglayer.readthedocs.io/)
- [Real-time EEG Examples](https://github.com/yourusername/real-time-eeg)
- [Performance Optimization Guide](https://real-time-eeg.readthedocs.io/) 
---
layout: post
title: "Applications of Deep Learning in EEG Signal Processing"
date: 2024-01-15
permalink: /blog/:year/:month/:title
description: "Exploring how to use deep learning models to process and analyze EEG signal data, including signal preprocessing, feature extraction, and classification methods."
tags: 
  - Deep Learning
  - EEG
  - Neural Networks
  - Signal Processing
author: jimjing
---

# Applications of Deep Learning in EEG Signal Processing

Deep learning has revolutionized how we process and analyze EEG signals. This article explores various deep learning approaches for EEG analysis.

## Introduction to Deep Learning for EEG

Deep learning models can automatically learn hierarchical features from raw EEG data, often outperforming traditional methods that rely on hand-crafted features.

## Convolutional Neural Networks for EEG

```python
import torch
import torch.nn as nn

class EEGConvNet(nn.Module):
    def __init__(self, n_channels=32, n_classes=4):
        super(EEGConvNet, self).__init__()
        
        # Temporal convolution
        self.temporal_conv = nn.Sequential(
            nn.Conv2d(1, 16, (1, 64), padding='same'),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # Spatial convolution
        self.spatial_conv = nn.Sequential(
            nn.Conv2d(16, 32, (n_channels, 1)),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 120)),
            nn.Flatten(),
            nn.Linear(32 * 120, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, n_classes)
        )
        
    def forward(self, x):
        # x shape: (batch, 1, channels, time)
        x = self.temporal_conv(x)
        x = self.spatial_conv(x)
        return self.classifier(x)
```

## Recurrent Neural Networks for EEG

```python
class EEGlstm(nn.Module):
    def __init__(self, input_size=32, hidden_size=64, num_layers=2, n_classes=4):
        super(EEGlstm, self).__init__()
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2,
            bidirectional=True
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_size, n_classes)
        )
        
    def forward(self, x):
        # x shape: (batch, time, channels)
        lstm_out, _ = self.lstm(x)
        # Use last time step output
        last_hidden = lstm_out[:, -1, :]
        return self.classifier(last_hidden)
```

## Transformer Models for EEG

```python
class EEGTransformer(nn.Module):
    def __init__(self, n_channels=32, n_timepoints=1000, n_classes=4):
        super(EEGTransformer, self).__init__()
        
        # Positional encoding
        self.pos_encoder = nn.Parameter(
            torch.randn(1, n_timepoints, n_channels)
        )
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=n_channels,
            nhead=8,
            dim_feedforward=128,
            dropout=0.1
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=4
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(n_channels * n_timepoints, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, n_classes)
        )
        
    def forward(self, x):
        # Add positional encoding
        x = x + self.pos_encoder
        # Transform
        x = self.transformer(x)
        # Classify
        x = x.reshape(x.size(0), -1)
        return self.classifier(x)
```

## Data Preprocessing for Deep Learning

```python
def preprocess_eeg(raw_eeg, sfreq=250):
    """Preprocess EEG data for deep learning"""
    # Bandpass filter
    filtered = mne.filter.filter_data(
        raw_eeg,
        sfreq=sfreq,
        l_freq=1,
        h_freq=40
    )
    
    # Normalize
    normalized = (filtered - filtered.mean(axis=1, keepdims=True)) / \
                 filtered.std(axis=1, keepdims=True)
    
    # Segment into epochs
    epochs = []
    window_size = int(2 * sfreq)  # 2-second windows
    stride = int(0.5 * sfreq)     # 0.5-second stride
    
    for i in range(0, normalized.shape[1] - window_size, stride):
        epochs.append(normalized[:, i:i+window_size])
    
    return np.array(epochs)
```

## Training Pipeline

```python
def train_eeg_model(model, train_loader, val_loader, epochs=100):
    """Train deep learning model on EEG data"""
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=5
    )
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        for data, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        # Validation
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for data, labels in val_loader:
                outputs = model(data)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
        # Update learning rate
        scheduler.step(val_loss)
        
        print(f'Epoch {epoch+1}/{epochs}:')
        print(f'Train Loss: {train_loss/len(train_loader):.4f}')
        print(f'Val Loss: {val_loss/len(val_loader):.4f}')
        print(f'Val Accuracy: {100.*correct/total:.2f}%')
```

## Best Practices

1. **Data Augmentation**
   - Time shifting
   - Adding noise
   - Channel dropout
   - Frequency warping

2. **Model Selection**
   - CNNs for spatial-temporal patterns
   - RNNs for sequential dependencies
   - Transformers for long-range dependencies

3. **Training Tips**
   - Use appropriate batch size
   - Apply learning rate scheduling
   - Implement early stopping
   - Use proper validation strategy

## Future Directions

1. **Self-supervised Learning**
   - Contrastive learning
   - Masked signal modeling
   - Signal reconstruction

2. **Transfer Learning**
   - Cross-subject adaptation
   - Cross-dataset generalization
   - Domain adaptation

3. **Interpretability**
   - Attention visualization
   - Layer-wise relevance propagation
   - Saliency mapping

## References

1. Craik, A., et al. (2019). "Deep learning for electroencephalogram (EEG) classification tasks: a review."
2. Roy, Y., et al. (2019). "Deep learning-based electroencephalography analysis: a systematic review."
3. Zhang, X., et al. (2021). "Deep learning for EEG-based brain-computer interfaces: A review." 
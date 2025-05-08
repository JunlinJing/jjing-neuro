---
layout: post
title: "EEG Data Analysis with Python: A Practical Guide"
date: 2024-02-20 10:00
image: /assets/images/blog/2024/eeg-python-tutorial.png
headerImage: true
tag:
- python
- neuroscience
- EEG
- tutorial
category: blog
author: jimjing
description: A step-by-step guide to analyzing EEG data using Python, from basic preprocessing to advanced machine learning applications
---

# EEG Data Analysis with Python: A Practical Guide

在这个教程中，我们将详细介绍如何使用Python进行脑电数据（EEG）分析。从数据加载、预处理到特征提取和机器学习应用，每个步骤都会提供详细的代码示例和解释。

## 环境准备

首先，我们需要设置Python环境并安装必要的包：

```bash
# 创建新的虚拟环境
python -m venv eeg_env
source eeg_env/bin/activate  # Linux/Mac
# 或
.\eeg_env\Scripts\activate  # Windows

# 安装依赖包
pip install mne==1.5.1 numpy==1.24.3 scipy==1.11.3 
pip install matplotlib==3.8.0 pandas==2.1.1 seaborn==0.13.0 
pip install scikit-learn==1.3.1
```

导入所需的库：

```python
import mne
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import signal
from sklearn.preprocessing import StandardScaler
import pandas as pd
import warnings
warnings.filterwarnings('ignore')  # 忽略警告信息

# 设置matplotlib中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False     # 用来正常显示负号
```

## 数据获取与加载

MNE-Python提供了示例数据集，我们可以用它来学习EEG分析流程：

```python
# 下载示例数据（如果尚未下载）
sample_data_folder = mne.datasets.sample.data_path()
raw_fname = sample_data_folder / 'MEG' / 'sample' / 'sample_audvis_raw.fif'

# 加载数据
raw = mne.io.read_raw_fif(raw_fname, preload=True)

# 只选择EEG通道
raw.pick_types(meg=False, eeg=True, eog=True)  # 保留EOG用于眼电伪迹检测

# 查看数据基本信息
print(f"数据采样率: {raw.info['sfreq']} Hz")
print(f"记录时长: {raw.times.max():.2f} 秒")
print(f"通道数量: {len(raw.ch_names)}")
```

## 数据预处理

### 1. 滤波处理

在EEG分析中，合适的滤波器设置至关重要：

```python
def apply_filters(raw_data, l_freq=1, h_freq=40, notch_freq=50):
    """
    应用滤波器处理EEG数据
    
    参数:
    - raw_data: MNE Raw对象
    - l_freq: 高通滤波截止频率
    - h_freq: 低通滤波截止频率
    - notch_freq: 陷波滤波器频率（用于去除电源干扰）
    """
    # 复制数据以免修改原始数据
    filtered_data = raw_data.copy()
    
    # 带通滤波
    filtered_data.filter(l_freq=l_freq, h_freq=h_freq, 
                        method='fir', phase='zero-double')
    
    # 陷波滤波（去除电源干扰）
    filtered_data.notch_filter(freqs=notch_freq, 
                             method='fir', phase='zero-double')
    
    return filtered_data

# 应用滤波器
raw_filtered = apply_filters(raw)

# 绘制滤波前后的频谱对比
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

raw.plot_psd(fmax=100, ax=ax1, show=False)
ax1.set_title('滤波前的频谱')

raw_filtered.plot_psd(fmax=100, ax=ax2, show=False)
ax2.set_title('滤波后的频谱')

plt.tight_layout()
plt.show()
```

### 2. 伪迹处理

EEG数据常见的伪迹包括眼电、肌电和电源干扰等：

```python
def remove_artifacts(raw_data, n_components=20, random_state=42):
    """
    使用ICA去除眼电和其他伪迹
    
    参数:
    - raw_data: MNE Raw对象
    - n_components: ICA组件数量
    - random_state: 随机种子
    """
    # 创建ICA对象
    ica = mne.preprocessing.ICA(n_components=n_components, 
                              random_state=random_state)
    
    # 应用ICA
    ica.fit(raw_data)
    
    # 自动检测眼电伪迹
    eog_indices, eog_scores = ica.find_bads_eog(raw_data)
    ica.exclude = eog_indices
    
    # 应用ICA去除伪迹
    cleaned_data = raw_data.copy()
    ica.apply(cleaned_data)
    
    # 绘制ICA组件
    ica.plot_components()
    
    # 绘制去伪迹前后的对比
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
    raw_data.plot(duration=5, n_channels=5, ax=ax1, show=False)
    ax1.set_title('原始数据')
    cleaned_data.plot(duration=5, n_channels=5, ax=ax2, show=False)
    ax2.set_title('去伪迹后的数据')
    plt.tight_layout()
    plt.show()
    
    return cleaned_data

# 应用伪迹去除
raw_cleaned = remove_artifacts(raw_filtered)
```

## 特征提取

### 1. 时频分析

我们可以计算不同频段的能量：

```python
def analyze_frequency_bands(data, sfreq, bands=None):
    """
    分析不同频段的能量
    
    参数:
    - data: EEG数据数组
    - sfreq: 采样率
    - bands: 频段定义字典
    """
    if bands is None:
        bands = {
            '德尔塔': (1, 4),
            '西塔': (4, 8),
            '阿尔法': (8, 13),
            '贝塔': (13, 30),
            '伽马': (30, 45)
        }
    
    results = {}
    for band_name, (fmin, fmax) in bands.items():
        # 计算功率谱密度
        freqs, psd = signal.welch(data, sfreq, nperseg=int(sfreq*2))
        
        # 提取频段能量
        idx = np.logical_and(freqs >= fmin, freqs <= fmax)
        band_power = np.mean(psd[:, idx], axis=1)
        
        results[band_name] = band_power
    
    # 绘制频段能量分布
    plt.figure(figsize=(10, 6))
    plt.boxplot(list(results.values()), labels=list(results.keys()))
    plt.title('各频段能量分布')
    plt.ylabel('能量 (µV²/Hz)')
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.show()
    
    return results

# 获取数据
data = raw_cleaned.get_data()
sfreq = raw_cleaned.info['sfreq']

# 分析频段能量
band_powers = analyze_frequency_bands(data, sfreq)
```

### 2. 时域特征

```python
def extract_time_features(data, sfreq):
    """
    提取时域特征
    """
    features = {
        'RMS': np.sqrt(np.mean(data**2, axis=1)),
        '峰峰值': np.ptp(data, axis=1),
        '方差': np.var(data, axis=1),
        '偏度': scipy.stats.skew(data, axis=1),
        '峰度': scipy.stats.kurtosis(data, axis=1)
    }
    
    # 创建特征DataFrame
    df_features = pd.DataFrame(features)
    
    # 显示统计信息
    print("\n时域特征统计信息:")
    print(df_features.describe())
    
    return df_features

# 提取时域特征
time_features = extract_time_features(data, sfreq)
```

## 机器学习应用

这里我们实现一个简单的EEG分类器：

```python
def build_eeg_classifier(X, y, test_size=0.2):
    """
    构建和评估EEG分类器
    
    参数:
    - X: 特征矩阵
    - y: 标签
    - test_size: 测试集比例
    """
    # 数据标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=test_size, random_state=42
    )
    
    # 创建分类器
    classifiers = {
        'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42),
        'SVM': SVC(kernel='rbf', random_state=42),
        'KNN': KNeighborsClassifier(n_neighbors=5)
    }
    
    # 训练和评估
    results = {}
    for name, clf in classifiers.items():
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        
        results[name] = {
            'accuracy': accuracy_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred, average='weighted'),
            'report': classification_report(y_test, y_pred)
        }
        
        print(f"\n{name} 分类结果:")
        print(results[name]['report'])
    
    return results

# 准备示例数据
# 这里我们使用之前提取的特征
features = np.column_stack([
    time_features,
    pd.DataFrame(band_powers)
])

# 生成示例标签（这里使用随机标签，实际应用中应使用真实标签）
labels = np.random.randint(0, 2, size=features.shape[0])

# 训练和评估分类器
classification_results = build_eeg_classifier(features, labels)
```

## 结论

本教程介绍了EEG数据分析的主要步骤：
1. 数据预处理（滤波和伪迹去除）
2. 特征提取（时频分析和时域特征）
3. 机器学习应用

完整代码和更多示例可在我们的[GitHub仓库](https://github.com/yourusername/eeg-python-tutorial)找到。

## 参考文献

1. Gramfort, A., et al. (2013). "MEG and EEG data analysis with MNE-Python"
2. Cohen, M. X. (2014). "Analyzing Neural Time Series Data"
3. Makeig, S., et al. (2004). "Mining event-related brain dynamics"
4. Lotte, F., et al. (2018). "A review of classification algorithms for EEG-based brain-computer interfaces" 
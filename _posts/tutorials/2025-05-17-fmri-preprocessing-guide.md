---
layout: post
title: "A Comprehensive Guide to fMRI Data Preprocessing"
description: "A step-by-step tutorial on preprocessing functional MRI data, covering motion correction, slice timing, spatial normalization, and more."
date: 2025-05-17
author: Jim Jing
categories: [tutorials]
tags: [fMRI, preprocessing, neuroimaging, tutorial]
---

# A Comprehensive Guide to fMRI Data Preprocessing

Preprocessing is a critical step in functional MRI analysis that aims to remove unwanted sources of variation and prepare the data for statistical analysis. This tutorial covers the essential preprocessing steps for fMRI data, with practical examples using popular neuroimaging software packages.

## Why Preprocessing Matters

Raw fMRI data contains various artifacts and noise sources that can obscure the neural signals of interest:

- Head motion during scanning
- Physiological noise (e.g., respiration, cardiac cycles)
- Scanner-related drift
- Timing differences between slice acquisitions
- Anatomical differences between subjects

Effective preprocessing minimizes these confounds, enhancing our ability to detect true neural activity and make valid inferences.

## Required Tools

For this tutorial, we'll use:
- **FSL** (FMRIB Software Library)
- **SPM** (Statistical Parametric Mapping)
- **Python** with the following libraries:
  - `nipype` (for workflow creation)
  - `nilearn` (for visualization)
  - `matplotlib` (for plotting)

## Step 1: DICOM to NIfTI Conversion

fMRI data is typically acquired in DICOM format but analyzed in NIfTI format. Here's how to convert:

```python
import os
from nipype.interfaces.dcm2nii import Dcm2niix

converter = Dcm2niix()
converter.inputs.source_dir = 'path/to/dicom_dir'
converter.inputs.output_dir = 'path/to/output_dir'
converter.inputs.compress = 'y'
converter.run()
```

## Step 2: Removing Initial Volumes

The first few volumes of an fMRI run are often discarded to allow for T1 equilibration effects:

```python
from nilearn import image
fmri_img = image.load_img('func.nii.gz')
n_vols_to_remove = 4
trimmed_img = image.index_img(fmri_img, slice(n_vols_to_remove, None))
trimmed_img.to_filename('func_trimmed.nii.gz')
```

## Step 3: Slice Timing Correction

Because fMRI volumes are acquired one slice at a time, different slices are actually acquired at different time points:

```python
from nipype.interfaces import spm
slice_timing = spm.SliceTiming()
slice_timing.inputs.in_files = 'func_trimmed.nii.gz'
slice_timing.inputs.num_slices = 36
slice_timing.inputs.time_repetition = 2.0
slice_timing.inputs.time_acquisition = 2.0 - (2.0/36)
slice_timing.inputs.slice_order = list(range(1, 37, 2)) + list(range(2, 37, 2))  # Interleaved acquisition
slice_timing.inputs.ref_slice = 1
slice_timing.run()
```

## Step 4: Motion Correction

Subject motion during scanning is one of the biggest sources of noise in fMRI:

```python
from nipype.interfaces import fsl
mcflirt = fsl.MCFLIRT()
mcflirt.inputs.in_file = 'slice_timing_corrected.nii.gz'
mcflirt.inputs.cost = 'mutualinfo'
mcflirt.inputs.ref_vol = 0
mcflirt.inputs.save_plots = True
mcflirt.run()
```

Visualizing motion parameters:

```python
import numpy as np
import matplotlib.pyplot as plt

motion_params = np.loadtxt('motion_parameters.par')
fig, ax = plt.subplots(2, 1, figsize=(12, 8))
ax[0].plot(motion_params[:, :3])
ax[0].set_ylabel('Translation (mm)')
ax[1].plot(motion_params[:, 3:])
ax[1].set_ylabel('Rotation (rad)')
ax[1].set_xlabel('Volume')
plt.tight_layout()
plt.savefig('motion_parameters.png')
```

## Step 5: Spatial Normalization

To compare brain activity across subjects, individual brains must be transformed to a standard space:

```python
# First, extract brain from structural image
bet = fsl.BET()
bet.inputs.in_file = 'structural.nii.gz'
bet.inputs.out_file = 'structural_brain.nii.gz'
bet.inputs.frac = 0.5
bet.run()

# Register functional to structural
flirt = fsl.FLIRT()
flirt.inputs.in_file = 'motion_corrected.nii.gz'
flirt.inputs.reference = 'structural_brain.nii.gz'
flirt.inputs.out_file = 'func2struct.nii.gz'
flirt.inputs.out_matrix_file = 'func2struct.mat'
flirt.run()

# Register structural to standard space (MNI)
flirt = fsl.FLIRT()
flirt.inputs.in_file = 'structural_brain.nii.gz'
flirt.inputs.reference = 'MNI152_T1_2mm_brain.nii.gz'
flirt.inputs.out_file = 'struct2mni.nii.gz'
flirt.inputs.out_matrix_file = 'struct2mni.mat'
flirt.run()

# Apply transformations to functional data
concat_xfm = fsl.ConvertXFM()
concat_xfm.inputs.in_file = 'func2struct.mat'
concat_xfm.inputs.in_file2 = 'struct2mni.mat'
concat_xfm.inputs.concat_xfm = True
concat_xfm.inputs.out_file = 'func2mni.mat'
concat_xfm.run()

apply_xfm = fsl.ApplyXFM()
apply_xfm.inputs.in_file = 'motion_corrected.nii.gz'
apply_xfm.inputs.reference = 'MNI152_T1_2mm_brain.nii.gz'
apply_xfm.inputs.in_matrix_file = 'func2mni.mat'
apply_xfm.inputs.out_file = 'func_mni.nii.gz'
apply_xfm.run()
```

## Step 6: Spatial Smoothing

Smoothing increases signal-to-noise ratio and accommodates anatomical variability:

```python
smooth = fsl.Smooth()
smooth.inputs.in_file = 'func_mni.nii.gz'
smooth.inputs.fwhm = 6.0
smooth.run()
```

## Step 7: Temporal Filtering

Remove low-frequency drifts and high-frequency noise:

```python
filt = fsl.TemporalFilter()
filt.inputs.in_file = 'smoothed_func.nii.gz'
filt.inputs.highpass_sigma = 50  # In seconds for 100s cutoff
filt.run()
```

## Step 8: Intensity Normalization

Scale voxel intensities to enable meaningful comparisons:

```python
from nilearn import image
img = image.load_img('filtered_func.nii.gz')
data = img.get_fdata()
mean = data.mean(axis=3)
data = data / mean[:, :, :, np.newaxis] * 100
norm_img = image.new_img_like(img, data)
norm_img.to_filename('normalized_func.nii.gz')
```

## Step 9: Quality Control

Always check your preprocessing results for anomalies:

```python
from nilearn import plotting
import matplotlib.pyplot as plt

# Create a report of mean, std, and tSNR images
img = image.load_img('normalized_func.nii.gz')
data = img.get_fdata()
mean_img = image.new_img_like(img, data.mean(axis=3))
std_img = image.new_img_like(img, data.std(axis=3))
tsnr_img = image.new_img_like(img, data.mean(axis=3) / data.std(axis=3))

fig, axes = plt.subplots(3, 1, figsize=(12, 15))
plotting.plot_epi(mean_img, axes=axes[0], title='Mean Image')
plotting.plot_epi(std_img, axes=axes[1], title='Standard Deviation')
plotting.plot_epi(tsnr_img, axes=axes[2], title='tSNR')
plt.tight_layout()
plt.savefig('quality_control.png')
```

## Common Preprocessing Workflows

Different research questions may require different preprocessing steps. Here are some common workflows:

1. **Task-based fMRI**:
   - Slice timing correction
   - Motion correction
   - Spatial normalization
   - Spatial smoothing (5-8mm FWHM)
   - Temporal filtering (high-pass, >0.01Hz)

2. **Resting-state fMRI**:
   - Motion correction
   - Regress out nuisance variables (CSF, WM signals, motion parameters)
   - Spatial normalization
   - Spatial smoothing (4-6mm FWHM)
   - Bandpass filtering (0.01-0.08Hz)

3. **Multi-echo fMRI**:
   - Combine echoes using optimal combination or ICA-based denoising
   - Motion correction
   - Spatial normalization
   - Minimal or no spatial smoothing

## Conclusion

Proper preprocessing is essential for reliable fMRI analysis. The choices made during preprocessing can significantly impact your results, so it's crucial to understand each step and make informed decisions based on your specific research question.

In future tutorials, we'll cover statistical analysis methods for extracting meaningful patterns from your preprocessed fMRI data.

## References

1. Poldrack, R. A., Mumford, J. A., & Nichols, T. E. (2011). Handbook of functional MRI data analysis. Cambridge University Press.
2. Jenkinson, M., Beckmann, C. F., Behrens, T. E., Woolrich, M. W., & Smith, S. M. (2012). FSL. Neuroimage, 62(2), 782-790.
3. Esteban, O., Markiewicz, C. J., Blair, R. W., et al. (2019). fMRIPrep: a robust preprocessing pipeline for functional MRI. Nature Methods, 16(1), 111-116. 
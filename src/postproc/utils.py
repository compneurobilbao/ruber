#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 17 11:23:57 2017

@author: asier
"""


def scrubbing(time_series, FD, thres=0.2):
    """
    simple scrubbing strategy based on timepoint removal
    """
    
    scrubbed_time_series = time_series.T[:,(FD<thres)]
    
    return scrubbed_time_series.T



def atlas_with_all_rois():
    atlas_old = '/home/asier/git/ruber/data/external/bha_atlas_2754_1mm_mni09c.nii.gz'
    atlas_new = opj(base_path, 'sub-001_atlas_2754_bold_space.nii.gz')

    atlas_new_img = nib.load(atlas_new)
    m = atlas_new_img.affine[:3, :3]

    atlas_old_data = nib.load(atlas_old).get_data()
    atlas_old_data_rois = np.unique(atlas_old_data)
    atlas_new_data = atlas_new_img.get_data()
    atlas_new_data_rois = np.unique(atlas_new_data)

    diff_rois = np.setdiff1d(atlas_old_data_rois, atlas_new_data_rois)

    for roi in diff_rois:
        p = np.argwhere(atlas_old_data == roi)[0]
        x, y, z = (np.round(np.diag(np.divide(p, m)))).astype(int)
        atlas_new_data[x, y, z] = roi

    atlas_new_data_img_corrected = nib.Nifti1Image(atlas_new_data,
                                                   affine=atlas_new_img.affine)
    nib.save(atlas_new_data_img_corrected,
             opj(base_path, 'sub-001_atlas_2754_bold_space.nii.gz'))
    
    









"""
from nilearn import datasets

dataset = datasets.fetch_atlas_harvard_oxford('cort-maxprob-thr25-2mm')
atlas_filename = dataset.maps
labels = dataset.labels

print('Atlas ROIs are located in nifti image (4D) at: %s' %
      atlas_filename)  # 4D data

# One subject of resting-state data
data = datasets.fetch_adhd(n_subjects=1)
fmri_filenames = data.func[0]


from nilearn.input_data import NiftiLabelsMasker
masker = NiftiLabelsMasker(labels_img=atlas_filename, standardize=True,
                           memory='nilearn_cache', verbose=5)

# Here we go from nifti files to the signal time series in a numpy
# array. Note how we give confounds to be regressed out during signal
# extraction
time_series = masker.fit_transform(fmri_filenames, confounds=data.confounds)

from nilearn.connectome import ConnectivityMeasure
correlation_measure = ConnectivityMeasure(kind='correlation')
correlation_matrix = correlation_measure.fit_transform([time_series_2754])[0]


# Plot the correlation matrix
import numpy as np
from matplotlib import pyplot as plt
plt.figure(figsize=(10, 10))
# Mask the main diagonal for visualization:
np.fill_diagonal(correlation_matrix, 0)

plt.imshow(correlation_matrix, interpolation="nearest", cmap="RdBu_r",
           vmax=0.8, vmin=-0.8)

# Add labels and adjust margins
plt.gca().yaxis.tick_right()
plt.subplots_adjust(left=.01, bottom=.3, top=.99, right=.62)  
"""
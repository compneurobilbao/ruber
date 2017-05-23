#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 17 11:34:08 2017

@author: asier
"""

# Nuisance regression





# location of experiment folder 
experiment_dir = '/home/asier/git/ruber'       
# list of subject identifiers         
subject_list = ['sub-001'] 

# TODO: move this to postproc.utils
from os.path import join as opj
from nilearn.input_data import NiftiLabelsMasker
from nilearn.image import resample_img
import nibabel as nib
from src.postproc.utils import scrubbing, locate_electrodes
import pandas as pd

base_path = '/home/asier/git/ruber/data/processed/fmriprep/sub-001/func/'

confounds = opj(base_path, 'sub-001_task-rest_bold_confounds.tsv')
preproc_data =  opj(base_path,  'sub-001_task-rest_bold_space-MNI152NLin2009cAsym_preproc.nii.gz')
confounds = pd.read_csv(confounds, delimiter='\t', na_values='n/a').fillna(0)

# to be build in subject space. 
# TODO: Check if this atlas is the same for all subjects (it should)
atlas_2514 = '/home/asier/git/ruber/data/external/bha_atlas_2514_1mm_mni09c.nii.gz'
atlas_2514_img = nib.load(atlas_2514)
fmri = nib.load(preproc_data)
resampled_2514_atlas = resample_img(atlas_2514_img, target_affine=fmri.affine,
                                    interpolation='nearest')
nib.save(resampled_2514_atlas, opj(base_path, 'sub-001_atlas_2514_bold_space.nii.gz'))

# to be build in subject space
atlas_2754 = '/home/asier/git/ruber/data/external/bha_atlas_2754_1mm_mni09c.nii.gz'
atlas_2754_img = nib.load(atlas_2754)
fmri = nib.load(preproc_data)
resampled_2754_atlas = resample_img(atlas_2754_img, target_affine=fmri.affine,
                                    interpolation='nearest')
nib.save(resampled_2754_atlas, opj(base_path, 'sub-001_atlas_2754_bold_space.nii.gz'))

# 1.- Nuisance regressors, filtering and ROI extraction with atlas
atlas_2514 = opj(base_path, 'sub-001_atlas_2514_bold_space.nii.gz')
atlas_2754 = opj(base_path, 'sub-001_atlas_2754_bold_space.nii.gz')

confounds_id = [ 'FramewiseDisplacement',
                'aCompCor0',
                'aCompCor1',
                'aCompCor2',
                'aCompCor3',
                'aCompCor4',
                'aCompCor5',
                'X',
                'Y',
                'Z',
                'RotX',
                'RotY',
                'RotZ',
                ]

confounds_matrix = confounds[confounds_id].as_matrix()

# atlas_2514
masker = NiftiLabelsMasker(labels_img=atlas_2514, background_label=0,
                           verbose=5, detrend=True, standardize=True, t_r=2.72,
                           smoothing_fwhm=6, low_pass=0.1, high_pass=0.01)

time_series_2514 = masker.fit_transform(preproc_data,
                                        confounds=confounds_matrix)

# atlas_2754
masker = NiftiLabelsMasker(labels_img=atlas_2754, background_label=0,
                           verbose=5, detrend=True, standardize=True, t_r=2.72,
                           smoothing_fwhm=6, low_pass=0.1, high_pass=0.01)

time_series_2754 = masker.fit_transform(preproc_data,
                                        confounds=confounds_matrix) 

# 2.- Scrubbing
# extract FramewiseDisplacement
FD = confounds.iloc[:,5].as_matrix()
thres = 0.2

time_series_2514 = scrubbing(time_series_2514, FD, thres)
time_series_2754 = scrubbing(time_series_2754, FD, thres)

np.savetxt(opj(base_path, 'time_series_2514.txt'), time_series_2514)
np.savetxt(opj(base_path, 'time_series_2754.txt'), time_series_2754)




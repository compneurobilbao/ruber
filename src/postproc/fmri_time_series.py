#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 17 11:34:08 2017

@author: asier
"""
from src.env import DATA, ATLAS_TYPES, CONFOUNDS_ID
from .utils import atlas_with_all_rois
import os.path as op
from os.path import join as opj

from nilearn.input_data import NiftiLabelsMasker
from nilearn.image import resample_img
import nibabel as nib
from src.postproc.utils import scrubbing
import pandas as pd
import numpy as np

PROCESSED = opj(DATA, 'processed')
EXTERNAL = opj(DATA, 'external')


def atlas_2_bold_space(sub, ses, atlas, preproc_data):

    atlas_path = opj(EXTERNAL, 'bha_' + atlas + '_1mm_mni09c.nii.gz')
    atlas_img = nib.load(atlas_path)
    fmri = nib.load(preproc_data)
    resampled_atlas = resample_img(atlas_img, target_affine=fmri.affine,
                                   interpolation='nearest')
    nib.save(resampled_atlas,
             opj(PROCESSED, 'fmriprep', sub, ses, 'func',
                 sub + '_' + ses + '_' + atlas + '_bold_space.nii.gz'))

    atlas_with_all_rois(sub, ses, atlas, opj(PROCESSED, 'fmriprep', sub, ses,
                                             'func',
                                             sub + '_' + ses + '_' +
                                             atlas + '_bold_space.nii.gz'))

    return opj(PROCESSED, 'fmriprep', sub, ses, 'func',
               sub + '_' + ses + '_' + atlas + '_bold_space.nii.gz')


def clean_and_get_time_series(subject_list, session_list):

    sub_ses_comb = [[subject, session] for subject in subject_list
                    for session in session_list]

    for sub, ses in sub_ses_comb:
        # TODO: CORRECT if exists
        #  not op.exists(op.join(PROCESSED, 'fmriprep', 'sub-' + sub,
#                                 'ses-' + ses))
        if True:
            print('Calculating: Subject ', sub, ' and session', ses)

            base_path = opj(PROCESSED, 'fmriprep', sub, ses, 'func')
            confounds_path = opj(base_path, sub + '_' + ses +
                                 '_task-rest_bold_confounds.tsv')
            preproc_data = opj(base_path, sub + '_' + ses +
                               '_task-rest_bold_space-MNI152NLin2009cAsym_preproc.nii.gz')

            confounds = pd.read_csv(confounds_path,
                                    delimiter='\t', na_values='n/a').fillna(0)

            for atlas in ATLAS_TYPES:
                atlas_path = atlas_2_bold_space(sub, ses, atlas, preproc_data)

                confounds_matrix = confounds[CONFOUNDS_ID].as_matrix()

                # atlas_2514
                masker = NiftiLabelsMasker(labels_img=atlas_path,
                                           background_label=0, verbose=5,
                                           detrend=True, standardize=True,
                                           t_r=2.72, smoothing_fwhm=6,
                                           # TR should not be a variable
                                           low_pass=0.1, high_pass=0.01)
                # 1.- Confound regression
                confounds_matrix = confounds[CONFOUNDS_ID].as_matrix()

                time_series = masker.fit_transform(preproc_data,
                                                   confounds=confounds_matrix)

                # 2.- Scrubbing
                # extract FramewiseDisplacement
                FD = confounds.iloc[:, 5].as_matrix()
                thres = 0.2
                time_series = scrubbing(time_series, FD, thres)

                # Save time series
                np.savetxt(opj(base_path, 'time_series_' + atlas + '.txt'),
                           time_series)
    return

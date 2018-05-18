# -*- coding: utf-8 -*-
from src.env import DATA, CONFOUNDS_ID

import os
from os.path import join as opj
import numpy as np

from src.postproc.utils import (load_elec_file,
                                execute,
                                scrubbing,
                                )
import nibabel as nib
from nilearn.image import resample_img

PROCESSED = opj(DATA, 'processed', 'fmriprep')
EXTERNAL = opj(DATA, 'external')
EXTERNAL_MNI_09c = opj(EXTERNAL, 'standard_mni_asym_09c')


def create_balloon(subject_list):
        
    for sub in subject_list:
       brain_mask_electrodes_to_09c(sub)
       centroid, rad = find_centroid_and_rad(sub)
       create_ball(sub, centroid, rad)
       remove_outliers(sub)
       identify_contacts_fmri(sub)
       
       
def extract_voxelwise_ts(subject_list, session_list):
    
    for sub in subject_list:
        identify_and_save_voxels(sub)  # identify, save info and create atlas
    
    extract_timeseries(subject_list, session_list)


def brain_mask_electrodes_to_09c(sub):

    ses = 'electrodes'

    """
    Brainmask electrodes space to 09c (and save omat)
    """
                   
    command = ['flirt',
               '-in',
               opj(DATA, 'raw', 'bids', sub, ses,
                   'electrodes_brain_mask.nii.gz'),
               '-ref',
               opj(EXTERNAL_MNI_09c,
                   'mni_icbm152_t1_tal_nlin_asym_09c_brain.nii'),
               '-out',
               opj(DATA, 'raw', 'bids', sub, ses,
                   'electrodes_brain_mask_09c.nii.gz'),
               '-init',
               opj(DATA, 'raw', 'bids', sub, ses,
                   'elec_2_09c_' + sub + '_' + ses + '.mat'),
               '-applyxfm', '-interp', 'nearestneighbour',
               ]


    for output in execute(command):
        print(output)


def identify_contacts_fmri(sub):
    
    rois_path = opj(DATA, 'raw', 'bids', sub, 'electrodes','ses-presurg',
                    'noatlas_3')
    
    for file in os.listdir(rois_path):
        file_path = opj(rois_path, file)
        data = nib.load(file_path).get_data()
        
        location = np.around(np.mean(np.asarray(np.where(data!=0)), axis=1))
        
        contact_name = file[4:-7]
        print(contact_name + ': ' + str(location))


def find_centroid_and_rad(sub):
    
    elec_file = opj(DATA, 'raw', 'bids', sub, 'electrodes',
                    'elec.loc')
    elec_location_mni09 = load_elec_file(elec_file)
    
    locations = [elec_location_mni09[key] for key in elec_location_mni09]
    centroid = np.mean(locations, axis=0)[0]
    
    distances = [np.sqrt((centroid[0]-loc[0][0])**2+
                         (centroid[1]-loc[0][1])**2+
                         (centroid[2]-loc[0][2])**2) for loc in locations]
        
    return centroid, np.max(distances)

    
def create_ball(sub, centroid, rad):
    
    import tempfile
    ses = 'electrodes'

    x, y, z = centroid.astype(int)
    temp_file = tempfile.mkstemp()

    # Create point
    command = ['fslmaths',
               opj(EXTERNAL_MNI_09c,
                   'mni_icbm152_t1_tal_nlin_asym_09c_brain.nii'),
               '-mul', '0', '-add', '1',
               '-roi', str(x), '1', str(y), '1', str(z), '1', '0', '1',
               temp_file[1],
               '-odt', 'float',
               ]
    for output in execute(command):
        print(output)

    sphere_size = int(rad)

    output_dir = opj(DATA, 'raw', 'bids', sub, ses, 'ses-presurg', 'balloon')

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Expand to sphere
    command = ['fslmaths',
               temp_file[1],
               '-kernel', 'sphere', str(sphere_size + 3),  # rad + 3 voxel
               '-fmeanu',
               temp_file[1],
               ]
    for output in execute(command):
        print(output)

    # Give value
    output_roi_path = opj(output_dir, 'balloon.nii.gz')
    command = ['fslmaths',
               temp_file[1],
               '-thr', str(1e-8), '-bin',
               output_roi_path,
               '-odt', 'float',
               ]
    for output in execute(command):
        print(output)


def remove_outliers(sub):
    from nilearn.masking import intersect_masks

    ses = 'electrodes'
    output_path = opj(DATA, 'raw', 'bids', sub, ses, 'ses-presurg', 'balloon',
                      'balloon_correct.nii.gz')

    balloon_path = opj(DATA, 'raw', 'bids', sub, ses, 'ses-presurg', 'balloon',
                       'balloon.nii.gz')
    
    mask_path = opj(DATA, 'raw', 'bids', sub, ses,
                   'electrodes_brain_mask_09c.nii.gz')
    
    balloon_img = nib.load(balloon_path)
    mask_img = nib.load(mask_path)
    
    intersected_img = intersect_masks([mask_img, balloon_img])
    
    # To BOLD space
    base_path = opj(PROCESSED, sub, 'ses-presurg', 'func')
    preproc_data = opj(base_path, sub + '_ses-presurg' 
                       '_task-rest_bold_space-MNI152NLin2009cAsym_preproc.nii.gz')
    fmri = nib.load(preproc_data)
    resampled_atlas = resample_img(intersected_img, target_affine=fmri.affine,
                                   interpolation='nearest')
    nib.save(resampled_atlas, output_path)


def identify_and_save_voxels(sub): # identify, save info and create atlas
    import json
    ses = 'electrodes'
    
    balloon_path = opj(DATA, 'raw', 'bids', sub, ses, 'ses-presurg', 'balloon',
                      'balloon_correct.nii.gz')
    output_path = opj(DATA, 'raw', 'bids', sub, ses, 'ses-presurg', 'balloon',
                      'balloon_atlas.nii.gz')
    output_json = opj(DATA, 'raw', 'bids', sub, ses, 'ses-presurg', 'balloon',
                      'loc_info.json')
    
    balloon_img = nib.load(balloon_path)
    affine = balloon_img.affine
    balloon_data = balloon_img.get_data()
    
    atlas_data = np.zeros(balloon_data.shape)
    
    idx = np.array(np.where(balloon_data == 1))
    
    for i in range(idx.shape[1]):
        atlas_data[idx[0,i], idx[1,i], idx[2,i]] = i+1
        
    atlas_img = nib.Nifti1Image(atlas_data,
                                affine=affine)
    nib.save(atlas_img, output_path)

    # loc info atlas
    loc_info = {i+1: [int(idx[0, i]),
                      int(idx[1, i]),
                      int(idx[2, i])] for i in range(idx.shape[1])}

    with open(output_json, 'w') as file:
        file.write(json.dumps(loc_info))
    
    
def extract_timeseries(subject_list, session_list):
    import pandas as pd
    from nilearn.input_data import NiftiLabelsMasker

    
    sub_ses_comb = [[subject, session] for subject in subject_list
                    for session in session_list]

    for sub, ses in sub_ses_comb:

        print('Calculating: Subject ', sub, ' and session', ses)

        base_path = opj(PROCESSED, sub, ses, 'func')
        confounds_path = opj(base_path, sub + '_' + ses +
                             '_task-rest_bold_confounds.tsv')
        preproc_data = opj(base_path, sub + '_' + ses +
                           '_task-rest_bold_space-MNI152NLin2009cAsym_preproc.nii.gz')

        confounds = pd.read_csv(confounds_path,
                                delimiter='\t', na_values='n/a').fillna(0)

    
        atlas_path = opj(DATA, 'raw', 'bids', sub, 'electrodes', 'ses-presurg', 'balloon',
                         'balloon_atlas.nii.gz')

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
        FD = confounds['FramewiseDisplacement'].as_matrix()
        thres = 0.2
        time_series = scrubbing(time_series, FD, thres)

        # Save time series
        np.savetxt(opj(base_path, 'time_series_balloon.txt'),
                   time_series)
    return
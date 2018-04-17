# -*- coding: utf-8 -*-
from src.env import DATA

import os
from os.path import join as opj

from src.postproc.utils import (load_elec_file,
                                execute,
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
       
       
def extract_voxelwise_ts(SUBJECT_LIST):
    
    for sub in subject_list:
        identify_and_save_voxels(sub)  # identify, save info and create atlas
        extract_timeseries(sub)


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
    
    
def extract_timeseries(sub):
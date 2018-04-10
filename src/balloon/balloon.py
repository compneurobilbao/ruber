# -*- coding: utf-8 -*-
from src.env import DATA

import os
import os.path as op
from os.path import join as opj
import numpy as np
import nibabel as nib
import subprocess

from src.postproc.utils import (load_elec_file,
                                order_dict,
                                execute,
                                )

PROCESSED = opj(DATA, 'processed', 'fmriprep')
EXTERNAL = opj(DATA, 'external')
EXTERNAL_MNI_09c = opj(EXTERNAL, 'standard_mni_asym_09c')


def create_balloon(subject_list):
        
    for sub in subject_list:
       brain_mask_electrodes_to_09c(sub)
       centroid, rad = find_centroid_and_rad(sub)
       create_ball(sub, centroid, rad)
       remove_outliers(sub)


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

    x, y, z = centroid
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

    sphere_size = rad

    output_dir = opj(DATA, 'raw', 'bids', sub, 'electrodes', ses,
                     'balloon')

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Expand to sphere
    command = ['fslmaths',
               temp_file[1],
               '-kernel', 'sphere', str(sphere_size + 3), # rad + 3 voxels
               '-fmean',
               temp_file[1],
               ]
    for output in execute(command):
        print(output)

    # Give value
    output_roi_path = opj(output_dir, 'balloon.nii.gz')
    command = ['fslmaths',
               temp_file[1],
               '-bin', '-mul', 1,
               output_roi_path,
               '-odt', 'float',
               ]
    for output in execute(command):
        print(output)
    

def remove_outliers(sub):
    
    
# -*- coding: utf-8 -*-
from src.env import DATA

import os
import os.path as op
from os.path import join as opj
import numpy as np
import nibabel as nib
import subprocess

from src.postproc.utils import load_elec_file, order_dict

PROCESSED = opj(DATA, 'processed', 'fmriprep')
EXTERNAL = opj(DATA, 'external')
EXTERNAL_MNI_09c = opj(EXTERNAL, 'standard_mni_asym_09c')


def execute(cmd):
    popen = subprocess.Popen(cmd,
                             stdout=subprocess.PIPE,
                             universal_newlines=True)
    for stdout_line in iter(popen.stdout.readline, ""):
        yield stdout_line
    popen.stdout.close()
    return_code = popen.wait()
    if return_code:
        raise subprocess.CalledProcessError(return_code, cmd)



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
    
    
    
def remove_outliers(sub):
    
    
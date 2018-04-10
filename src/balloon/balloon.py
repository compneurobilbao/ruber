# -*- coding: utf-8 -*-
from src.env import DATA, ATLAS_TYPES, NEIGHBOURS, ELECTRODE_SPHERE_SIZE

import os
import os.path as op
from os.path import join as opj
import numpy as np
import nibabel as nib
import subprocess

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
       brain_mask_electrodes_to_09c(subject_list)
       find_centroid(subject_list)
       create_ball(subject_list)
       remove_outliers(subject_list)


def brain_mask_electrodes_to_09c(subject_list):

    ses = 'electrodes'

    for sub in subject_list:
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




def find_centroid(subject_list):
    

    
    
def create_ball(subject_list):
    
    
    
def remove_outliers(subject_list):
    
    
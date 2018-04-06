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


def atlas_to_t1(subject_list, session_list):
    """
    Atlas to T1w space
    """
    sub_ses_comb = [[subject, session] for subject in subject_list
                    for session in session_list]

    for sub, ses in sub_ses_comb:

        print('Calculating: Subject ', sub, ' and session', ses)

        # Extract brain from subject space
        command = ['fslmaths',
                   opj(PROCESSED, sub, ses, 'anat',
                       sub + '_' + ses + '_T1w_preproc.nii.gz'),
                   '-mas',
                   opj(PROCESSED, sub, ses, 'anat',
                       sub + '_' + ses + '_T1w_brainmask.nii.gz'),
                   opj(PROCESSED, sub, ses, 'anat',
                       sub + '_' + ses + '_T1w_brain.nii.gz'),
                   ]
        for output in execute(command):
            print(output)

        # Brain 09c -> Brain subject (save omat)
        command = ['flirt',
                   '-in',
                   opj(EXTERNAL_MNI_09c,
                       'mni_icbm152_t1_tal_nlin_asym_09c_brain.nii'),
                   '-ref',
                   opj(PROCESSED, sub, ses, 'anat',
                       sub + '_' + ses + '_T1w_brain.nii.gz'),
                   '-omat',
                   opj(PROCESSED, sub, ses, 'anat',
                       '09c_2_' + sub + '_' + ses + '.mat'),
                   ]
        for output in execute(command):
            print(output)

        for atlas in ATLAS_TYPES:
            # Atlas 09c -> Subject space (using previous omat)
            command = ['flirt',
                       '-in',
                       opj(EXTERNAL,
                           'bha_' + atlas + '_1mm_mni09c.nii.gz'),
                       '-ref',
                       opj(PROCESSED, sub, ses, 'anat',
                           sub + '_' + ses + '_T1w_brain.nii.gz'),
                       '-out',
                       opj(PROCESSED, sub, ses, 'anat',
                           sub + '_' + ses + '_' + atlas + '.nii.gz'),
                       '-init',
                       opj(PROCESSED, sub, ses, 'anat',
                           '09c_2_' + sub + '_' + ses + '.mat'),
                       '-applyxfm', '-interp', 'nearestneighbour',
                       ]
            for output in execute(command):
                print(output)
            atlas_with_all_rois(sub, ses, atlas, opj(PROCESSED, sub, ses,
                                                     'anat', sub + '_' +
                                                     ses + '_' + atlas +
                                                     '.nii.gz'))

    return




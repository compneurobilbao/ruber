#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 25 11:35:55 2017

@author: asier
"""
from src.env import BIDS_DATA, DATA
import shutil
import os
from os.path import join as opj
import json
import subprocess


"""
fmriprep
"""

DATA_DIR = BIDS_DATA
OUTPUT_DIR = opj(DATA, 'processed')
WORK_DIR = opj(DATA, 'interim')

docker run -ti --rm \
	-v $DATA_DIR:/data:ro \
	-v $OUTPUT_DIR:/output \
	-v $WORK_DIR:/work \
	-w /work \
	poldracklab/fmriprep:latest \
	/data /output participant --participant_label sub-001 \
	-w /work --no-freesurfer --ignore fieldmaps \
     --output-space template --template MNI152NLin2009cAsym



"""
MRIQC
"""

DATA_DIR = BIDS_DATA
OUTPUT_DIR = opj(DATA, 'processed')
WORK_DIR = opj(DATA, 'interim')


docker run -ti --rm \
	-v $DATA_DIR:/data:ro \
	-v $OUTPUT_DIR:/output \
	-v $WORK_DIR:/work \
	-w /work \
	poldracklab/mriqc:latest \
	/data /output participant --participant_label sub-001 \
	-w /work --verbose-reports

sudo chmod 777 -R $DATA



T1='/home/asier/Desktop/test_ruber/t1.nii.gz'
${FSLDIR}/bin/bet $T1 T1_brain -B -f "0.1" -s -m 
 
flirt -in atlas_3000.nii -ref ${FSLDIR}/data/standard/MNI152_T1_1mm_brain_mask.nii.gz -out atlas_1000 -cost mutualinfo -interp nearestneighbour
flirt -in T1_brain -ref  ${FSLDIR}/data/standard/MNI152_T1_1mm_brain.nii.gz -out t1_brain_mni

                          
flirt -in /home/asier/Desktop/test_ruber/atlas_3000.nii -ref /home/asier/Desktop/mni_icbm152_nlin_asym_09c_nifti/mni_icbm152_nlin_asym_09c/mni_icbm152_t1_tal_nlin_asym_09c_mask.nii -out /home/asier/Desktop/bha_atlas_1mm_mni09c.nii.gz -cost mutualinfo -interp nearestneighbour
                         
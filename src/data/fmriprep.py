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
    --output-space template --template MNI152NLin2009cAsymZ

    
#  T1w


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



"""
Atlas to 2009c
"""
## 3mm to 1mm
flirt -in /home/asier/Desktop/test_ruber/atlas_2514.nii.gz \
-ref  /home/asier/Desktop/test_ruber/MNI152_T1_3mm_brain.nii.gz \
-out /home/asier/Desktop/test_ruber/atlas_2514_1mm.nii.gz \
-init /home/asier/Desktop/test_ruber/3mmto1mm.mat -applyxfm -interp nearestneighbour 

flirt -in /home/asier/Desktop/test_ruber/atlas_2754.nii.gz \
-ref  /home/asier/Desktop/test_ruber/MNI152_T1_3mm_brain.nii.gz \
-out /home/asier/Desktop/test_ruber/atlas_2754_1mm.nii.gz \
-init /home/asier/Desktop/test_ruber/3mmto1mm.mat -applyxfm -interp nearestneighbour 

## Brain MNI152 -> Brain 2009c (save omat) 
## Atlas MNI152 -> 2009c space (using previous omat)

flirt -in ${FSLDIR}/data/standard/MNI152_T1_1mm_brain.nii.gz  \
-ref /home/asier/git/ruber/data/external/standard_mni_asym_09c/mni_icbm152_t1_tal_nlin_asym_09c_brain.nii \
-cost mutualinfo -omat mni152_2_09c.mat


flirt -in /home/asier/Desktop/test_ruber/atlas_2514_1mm.nii.gz \
-ref /home/asier/git/ruber/data/external/standard_mni_asym_09c/mni_icbm152_t1_tal_nlin_asym_09c_brain.nii \
-out /home/asier/git/ruber/data/external/bha_atlas_2514_1mm_mni09c.nii.gz \
-init /home/asier/git/ruber/data/external/standard_mni_asym_09c/mni152_2_09c.mat -applyxfm -interp nearestneighbour 

flirt -in /home/asier/Desktop/test_ruber/atlas_2754_1mm.nii.gz \
-ref /home/asier/git/ruber/data/external/standard_mni_asym_09c/mni_icbm152_t1_tal_nlin_asym_09c_brain.nii \
-out /home/asier/git/ruber/data/external/bha_atlas_2754_1mm_mni09c.nii.gz \
-init /home/asier/git/ruber/data/external/standard_mni_asym_09c/mni152_2_09c.mat -applyxfm -interp nearestneighbour 


"""
Extract brain from electrodes T1W
"""

T1='/home/asier/Desktop/test_ruber/t1.nii.gz'
${FSLDIR}/bin/bet $T1 T1_brain -B -f "0.1" -s -m 

"""
Atlas to subject space
"""
                         
flirt -in T1_brain -ref   -out t1_brain_mni                      





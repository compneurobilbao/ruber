#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 25 11:35:55 2017

@author: asier
"""
from src.env import BIDS_DATA, DATA, NTHREADS
import shutil
import os.path as op
from os.path import join as opj
import json
import subprocess

DATA_DIR = BIDS_DATA
OUTPUT_DIR = opj(DATA, 'processed')
WORK_DIR = opj(DATA, 'interim')


def run_fmriprep(subject_list, session_list):

    sub_ses_comb = [[subject, session] for subject in subject_list
                    for session in session_list]

    for sub, ses in sub_ses_comb:
        if not op.exists(op.join(OUTPUT_DIR, 'fmriprep', 'sub-' + sub,
                                 'ses-' + ses)):
            print('Calculating: Subject ', sub, ' and session', ses)

            command = [
                   'docker', 'run', '-i', '--rm',
                   '-v', DATA_DIR + ':/data:ro',
                   '-v', OUTPUT_DIR + ':/output',
                   '-v', WORK_DIR + ':/work',
                   '-w', '/work',
                   'poldracklab/fmriprep:latest',
                   '/data', '/output', 'participant',
                   '--participant_label', sub, '-s', ses,
                   '-w', '/work', '--no-freesurfer', '--ignore', 'fieldmaps',
                   '--n_cpus', str(NTHREADS),
                   '--output-space', 'template',
                   '--template', 'MNI152NLin2009cAsym',
                ]
            
            output, error = subprocess.Popen(
                                    command, universal_newlines=True,
                                    stdout=subprocess.PIPE,
                                    stderr=subprocess.PIPE).communicate()
    

def run_mriqc(subject_list, session_list):
    
    
"""
fmriprep
"""

docker run -ti --rm \
    -v $DATA_DIR:/data:ro \
    -v $OUTPUT_DIR:/output \
    -v $WORK_DIR:/work \
    -w /work \
    poldracklab/fmriprep:latest \
    /data /output participant --participant_label sub-001 \
    --session-id post \
    -w /work --no-freesurfer --ignore fieldmaps \
    --output-space template --template MNI152NLin2009cAsym

    

"""
MRIQC
"""


docker run -ti --rm \
	-v $DATA_DIR:/data:ro \
	-v $OUTPUT_DIR:/output \
	-v $WORK_DIR:/work \
	-w /work \
	poldracklab/mriqc:latest \
	/data /output participant --participant_label sub-001 \
	-w /work --verbose-reports

sudo chmod 777 -R $DATA





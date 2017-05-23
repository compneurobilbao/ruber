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
            
            for output in execute(command):
                print(output)
    

def run_mriqc(subject_list, session_list):
    
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
                   'poldracklab/mriqc:latest',
                   '/data', '/output', 'participant',
                   '--participant_label', sub, '-s', ses,
                   '-w', '/work', '--verbose-reports',
                ]
            
            for output in execute(command):
                print(output)
    

# sudo chmod 777 -R $DATA





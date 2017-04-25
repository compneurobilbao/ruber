#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 25 11:35:55 2017

@author: asier
"""
from src.env import BIDS_DATA, DATA
import shutil
import os
import os.path as op
import json
import subprocess


"""
fmriprep
"""

DATA_DIR = BIDS_DATA
OUTPUT_DIR = op.join(DATA, 'processed')
WORK_DIR = op.join(DATA, 'interim')

docker run -ti --rm \
	-v $DATA_DIR:/data:ro \
	-v $OUTPUT_DIR:/output \
	-v $WORK_DIR:/work \
	-w /work \
	poldracklab/fmriprep:latest \
	/data /output participant --participant_label sub-001 \
	-w /work --no-freesurfer --ignore fieldmaps


"""
MRIQC
"""

DATA_DIR = BIDS_DATA
OUTPUT_DIR = op.join(DATA, 'processed')
WORK_DIR = op.join(DATA, 'interim')


docker run -ti --rm \
	-v $DATA_DIR:/data:ro \
	-v $OUTPUT_DIR:/output \
	-v $WORK_DIR:/work \
	-w /work \
	poldracklab/mriqc:latest \
	/data /output participant --participant_label sub-001 \
	-w /work --verbose-reports

sudo chmod 777 -R DATA_DIR






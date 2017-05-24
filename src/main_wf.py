#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from src.preproc import run_fmriprep, run_mriqc

from src.postproc.utils import atlas_to_t1

from src.dmri import run_dti_artifact_correction
from src.dmri import run_spm_fsl_dti_preprocessing
from src.dmri import run_camino_tractography

from src.postproc.fmri_time_series import clean_and_get_time_series

from src.postproc.utils import t1w_electrodes_to_09c, locate_electrodes


subject_list = ['sub-001']
session_list = ['ses-presurg']

"""
fmriprep and mriqc calls
"""


run_fmriprep(subject_list, session_list)

run_mriqc(subject_list, session_list)

# WARNING!! Execute permission change over files before continue
# sudo chmod d------rwx -R $OUTPUT_DIR
# sudo chmod 777 -R $OUTPUT_DIR


"""
Atlas to T1w space
"""

atlas_to_t1(subject_list, session_list)

"""
dMRI pipeline
"""

run_dti_artifact_correction(subject_list, session_list)

run_spm_fsl_dti_preprocessing(subject_list, session_list)

run_camino_tractography(subject_list, session_list)

# Visualization
# http://web4.cs.ucl.ac.uk/research/medic/camino/pmwiki/pmwiki.php?n=Tutorials.TrackingTutorial


"""
fMRI pipeline postproc
"""
# WARINING! Create elec file for each subject manually !!
# FIRST: T1w_electrodes to 09c space

t1w_electrodes_to_09c(subject_list)

clean_and_get_time_series(subject_list)

                  

# TODO: Include this in the pipeline in an organize way, preferrably following BIDS



    
    
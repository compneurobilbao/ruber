#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from src.preproc import run_fmriprep, run_mriqc

from src.postproc.utils import atlas_to_t1

from src.dmri import run_dti_artifact_correction
from src.dmri import run_spm_fsl_dti_preprocessing
from src.dmri import run_camino_tractography

from src.postproc.fmri_time_series import clean_and_get_time_series

from src.postproc.utils import (t1w_electrodes_to_09c,
                                locate_electrodes,
                                locate_electrodes_closest_roi,
                                )

SUBJECT_LIST = ['sub-001']
SESSION_LIST = ['ses-presurg']

"""
fmriprep and mriqc calls
"""

run_fmriprep(SUBJECT_LIST, SESSION_LIST)

run_mriqc(SUBJECT_LIST, SESSION_LIST)

# WARNING!! Execute permission change over files before continue
# sudo chmod d------rwx -R $OUTPUT_DIR
# sudo chmod 777 -R $OUTPUT_DIR


"""
Atlas to T1w space
"""

atlas_to_t1(SUBJECT_LIST, SESSION_LIST)

"""
dMRI pipeline
"""

run_dti_artifact_correction(SUBJECT_LIST, SESSION_LIST)

run_spm_fsl_dti_preprocessing(SUBJECT_LIST, SESSION_LIST)

run_camino_tractography(SUBJECT_LIST, SESSION_LIST)

# Visualization
# http://web4.cs.ucl.ac.uk/research/medic/camino/pmwiki/pmwiki.php?n=Tutorials.TrackingTutorial


"""
fMRI pipeline postproc
"""

clean_and_get_time_series(SUBJECT_LIST, SESSION_LIST)

"""
Electrodes location pipeline (WARNING: Some manual work)
"""

# FIRST: T1w_electrodes to 09c space

t1w_electrodes_to_09c(SUBJECT_LIST)

# WARNING! Create elec file for each subject manually !!
# from src.postproc.utils import contacts_from_electrode
# elec_name = 'OIM1'
# contact_num = 12
# first_contact_pos = [78, 73, 78]
# last_contact_pos = [80, 26, 84]
# contacts_from_electrode(first_contact_pos, last_contact_pos, contact_num, elec_name)

locate_electrodes(SUBJECT_LIST)

# or

locate_electrodes_closest_roi(SUBJECT_LIST)
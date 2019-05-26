#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# you have to be in the ruber directory in the terminal befiore entering in python
from src.preproc import run_fmriprep, run_mriqc

from src.postproc.utils import elec_to_t1

from src.dmri import (run_dti_artifact_correction,
                      run_spm_fsl_dti_preprocessing,
                      run_camino_tractography,
                      run_dtk_tractography,
                      )

from src.dmri.utils import (correct_dwi_space_atlas,
                            get_con_matrix_matlab,
                            )

from src.postproc.fmri_time_series import clean_and_get_time_series

from src.postproc.utils import (t1w_electrodes_to_09c,
                                locate_electrodes,
                                locate_electrodes_closest_roi,
                                calc_con_mat_electrodes_noatlas,
                                export_data_to_mat,
                                )

SUBJECT_LIST = ['sub-010']
SESSION_LIST = ['ses-presurg']

"""
fmriprep and mriqc calls
"""

run_fmriprep(SUBJECT_LIST, SESSION_LIST)

run_mriqc(SUBJECT_LIST, SESSION_LIST)

# WARNING!! Execute permission change over files before continue; so go to terminal by typing "exit()" and themn run the two commands below, then go back to python
#OUTPUT_DIR=("/home/asier/git/ruber/data/processed")
#sudo chmod 777 -R $OUTPUT_DIR


"""
Atlas to T1w space
"""

t1w_electrodes_to_09c(SUBJECT_LIST) 
elec_to_t1(SUBJECT_LIST, SESSION_LIST)

"""
fMRI pipeline postproc
"""

clean_and_get_time_series(SUBJECT_LIST, SESSION_LIST)


"""
dMRI pipeline
"""

# This correction might not be needed if we already run it!!
run_dti_artifact_correction(SUBJECT_LIST, SESSION_LIST)

run_spm_fsl_dti_preprocessing(SUBJECT_LIST, SESSION_LIST)

correct_dwi_space_atlas(SUBJECT_LIST, SESSION_LIST)

run_camino_tractography(SUBJECT_LIST, SESSION_LIST)

# This correction might not be needed if we already run it!!
# run_dtk_tractography(SUBJECT_LIST, SESSION_LIST)

get_con_matrix_matlab(SUBJECT_LIST, SESSION_LIST)

# Visualization
# http://web4.cs.ucl.ac.uk/research/medic/camino/pmwiki/pmwiki.php?n=Tutorials.TrackingTutorial


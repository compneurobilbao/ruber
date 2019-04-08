#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# you have to be in the ruber directory in the terminal befiore entering in python
from src.preproc import run_fmriprep, run_mriqc

from src.postproc.utils import atlas_to_t1

from src.dmri import (run_dti_artifact_correction,
                      run_spm_fsl_dti_preprocessing,
                      run_camino_tractography,
                      run_dtk_tractography,
                      )
from src.dmri.utils import correct_dwi_space_atlas

from src.postproc.fmri_time_series import clean_and_get_time_series_noatlas

from src.postproc.utils import (t1w_electrodes_to_09c,
                                locate_electrodes,
                                locate_electrodes_closest_roi,
                                calc_con_mat_electrodes_noatlas
                                )

SUBJECT_LIST = ['sub-001']
SESSION_LIST = ['ses-presurg']

## WARNING: FIRST RUN atlas_wf.py

"""
Electrodes location pipeline (WARNING: Some manual work)
"""

# FIRST: T1w_electrodes to 09c space
# TODO: This part to nipype
t1w_electrodes_to_09c(SUBJECT_LIST) # Gianna: check if this is necessary!

# WARNING! Create elec file (/home/asier/git/ruber/data/raw/bids/sub-XXX/electrodes/elec.loc)
# manually !! Use electrodes_brain_09c.nii.gz
# NOTE that in the common t1 space, I created the file elecT1space in the folder /home/asier/git/ruber/data/processed/fmriprep/sub-XXX/ses-presurg/anat/  
#loc
from src.postproc.utils import contacts_from_electrode
elec_name = 'H'
contact_num = 12
first_contact_pos = [64, 95, 145]
last_contact_pos = [ 34, 71, 141]
contacts_from_electrode(first_contact_pos, last_contact_pos, contact_num, elec_name)

# copy the result from 'contacts_from_electrode' to elec.loc


"""
APPROACH 2: NOATLAS
"""
calc_con_mat_electrodes_noatlas(SUBJECT_LIST, SESSION_LIST)

"""
fMRI pipeline postproc
"""
clean_and_get_time_series_noatlas(SUBJECT_LIST, SESSION_LIST)

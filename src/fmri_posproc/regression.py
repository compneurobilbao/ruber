#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 17 11:34:08 2017

@author: asier
"""

# Nuisance regression
















# Filtering (probably with nuisance regression using signal.clean from nilearn)



# Include 6 motion parameters, FD, and aCompCor
import pandas as pd
confounds = '/home/asier/git/ruber/data/processed/fmriprep/sub-001/func/sub-001_task-rest_bold_confounds.tsv'
'/home/asier/git/ruber/data/processed/fmriprep/sub-001/func/sub-001_task-rest_bold_space-MNI152NLin2009cAsym_preproc.nii.gz'
data = pd.read_csv(confounds, delimiter='\t', na_values='n/a').fillna(0)

data['vx-wisestdDVARS']

'FramewiseDisplacement'
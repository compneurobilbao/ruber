#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Extract brain from 09c (just once)
"""

T1='/home/asier/git/ruber/data/external/standard_mni_asym_09c/mni_icbm152_t1_tal_nlin_asym_09c.nii'
mask='/home/asier/git/ruber/data/external/standard_mni_asym_09c/mni_icbm152_t1_tal_nlin_asym_09c_mask.nii'
fslmaths $T1 -mas $mask mni_icbm152_t1_tal_nlin_asym_09c_brain.nii

"""
Atlas to T1W space
"""
#Extract brain from subject space
fslmaths /home/asier/git/ruber/data/processed/fmriprep/sub-001/anat/sub-001_T1w_preproc.nii.gz \
-mas /home/asier/git/ruber/data/processed/fmriprep/sub-001/anat/sub-001_T1w_brainmask.nii.gz \
/home/asier/git/ruber/data/processed/fmriprep/sub-001/anat/sub-001_T1w_brain.nii.gz

## this must be included in the pipelines
## Brain 09c -> Brain subject (save omat) (previously we have to generate Brain Subject)
## Atlas 09c -> Subject space (using previous omat)
flirt -in /home/asier/git/ruber/data/external/standard_mni_asym_09c/mni_icbm152_t1_tal_nlin_asym_09c_brain.nii \
-ref /home/asier/git/ruber/data/processed/fmriprep/sub-001/anat/sub-001_T1w_brain.nii.gz \
-omat /home/asier/git/ruber/data/processed/fmriprep/sub-001/anat/09c_2_sub-001.mat

flirt -in  /home/asier/git/ruber/data/external/bha_atlas_2514_1mm_mni09c.nii.gz \
-ref /home/asier/git/ruber/data/processed/fmriprep/sub-001/anat/sub-001_T1w_brain.nii.gz \
-out /home/asier/git/ruber/data/processed/fmriprep/sub-001/anat/sub-001_atlas_2514.nii.gz \
-init /home/asier/git/ruber/data/processed/fmriprep/sub-001/anat/09c_2_sub-001.mat \
-applyxfm -interp nearestneighbour 

flirt -in  /home/asier/git/ruber/data/external/bha_atlas_2754_1mm_mni09c.nii.gz \
-ref /home/asier/git/ruber/data/processed/fmriprep/sub-001/anat/sub-001_T1w_brain.nii.gz \
-out /home/asier/git/ruber/data/processed/fmriprep/sub-001/anat/sub-001_atlas_2754.nii.gz \
-init /home/asier/git/ruber/data/processed/fmriprep/sub-001/anat/09c_2_sub-001.mat \
-applyxfm -interp nearestneighbour 





"""
dMRI pipeline
"""

# location of experiment folder 
experiment_dir = '/home/asier/git/ruber'       
# list of subject identifiers         
subject_list = ['sub-001']    
    
from src.dmri import run_dti_artifact_correction
run_dti_artifact_correction(experiment_dir, subject_list)

from src.dmri import run_spm_fsl_dti_preprocessing
run_spm_fsl_dti_preprocessing('/home/asier/git/ruber', ['sub-001'])

from src.dmri import run_camino_tractography
run_camino_tractography('/home/asier/git/ruber', ['sub-001'] )

http://web4.cs.ucl.ac.uk/research/medic/camino/pmwiki/pmwiki.php?n=Tutorials.TrackingTutorial


"""
fMRI nuisance pipeline
"""
# Include 6 motion parameters, FD, and aCompCor
import pandas as pd
confounds = '/home/asier/git/ruber/data/processed/fmriprep/sub-001/func/sub-001_task-rest_bold_confounds.tsv'
'/home/asier/git/ruber/data/processed/fmriprep/sub-001/func/sub-001_task-rest_bold_space-MNI152NLin2009cAsym_preproc.nii.gz'
data = pd.read_csv(confounds, delimiter='\t', na_values='n/a').fillna(0)

data['vx-wisestdDVARS']

'FramewiseDisplacement'
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
fMRI pipeline postproc
"""

from os.path import join as opj
from nilearn.input_data import NiftiLabelsMasker
from nilearn.image import resample_img
import nibabel as nib
from src.fmri_posproc import scrubbing
import pandas as pd

base_path = '/home/asier/git/ruber/data/processed/fmriprep/sub-001/func/'

confounds = opj(base_path, 'sub-001_task-rest_bold_confounds.tsv')
preproc_data =  opj(base_path,  'sub-001_task-rest_bold_space-MNI152NLin2009cAsym_preproc.nii.gz')
confounds = pd.read_csv(confounds, delimiter='\t', na_values='n/a').fillna(0)

# to be build in subject space
atlas_2514 = '/home/asier/git/ruber/data/external/bha_atlas_2514_1mm_mni09c.nii.gz'
atlas_2514_img = nib.load(atlas_2514)
fmri = nib.load(preproc_data)
resampled_2514_atlas = resample_img(atlas_2514_img, target_affine=fmri.affine)
nib.save(resampled_2514_atlas, opj(base_path, 'sub-001_atlas_2514_bold_space.nii.gz'))

# to be build in subject space
atlas_2754 = '/home/asier/git/ruber/data/external/bha_atlas_2754_1mm_mni09c.nii.gz'
atlas_2754_img = nib.load(atlas_2754)
fmri = nib.load(preproc_data)
resampled_2754_atlas = resample_img(atlas_2754_img, target_affine=fmri.affine)
nib.save(resampled_2754_atlas, opj(base_path, 'sub-001_atlas_2754_bold_space.nii.gz'))


# 1.- Nuisance regressors, filtering and ROI extraction with atlas
atlas_2514 = opj(base_path, 'sub-001_atlas_2514_bold_space.nii.gz')
atlas_2754 = opj(base_path, 'sub-001_atlas_2754_bold_space.nii.gz')

confounds_id = [ 'FramewiseDisplacement',
                'aCompCor0',
                'aCompCor1',
                'aCompCor2',
                'aCompCor3',
                'aCompCor4',
                'aCompCor5',
                'X',
                'Y',
                'Z',
                'RotX',
                'RotY',
                'RotZ',
                ]

# standardize=False?¿
masker = NiftiLabelsMasker(labels_img=atlas_2514, detrend=True, standardize=True,
                     smoothing_fwhm=6,
                     low_pass=0.1, high_pass=0.01)

time_series = masker.fit_transform(preproc_data,
                                   confounds=confounds[confounds_id].as_matrix()) # confounds etc. here?


# data_img = nibabel.Nifti1Image(time_series, fmri.affine)

# 2.- Scrubbing

# extract six movement/motion parameters into rest_mc.1D
confounds.iloc[:,-6:].to_csv(opj(base_path,'rest_mc.1D'), sep='\t', header=False, index=False)
# extract FramewiseDisplacement into frames_in.1D
confounds.iloc[:,5].to_csv(opj(base_path,'frames_in.1D'), sep='\t', header=False, index=False)


sc = scrubbing.create_scrubbing_preproc()
sc.inputs.inputspec.frames_in_ID = 'frames_in.1D'
sc.inputs.inputpsec.movement_parameters = 'rest_mc.1D'
sc.inputs.inputpsec.preprocessed = preproc_data
sc.run()









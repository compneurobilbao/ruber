#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from src.preproc import run_fmriprep, run_mriqc

from src.postproc.utils import atlas_to_t1

from src.dmri import run_dti_artifact_correction
from src.dmri import run_spm_fsl_dti_preprocessing
from src.dmri import run_camino_tractography

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

# location of experiment folder 
experiment_dir = '/home/asier/git/ruber'       
# list of subject identifiers         
subject_list = ['sub-001'] 

# TODO: move this to postproc.utils
from os.path import join as opj
from nilearn.input_data import NiftiLabelsMasker
from nilearn.image import resample_img
import nibabel as nib
from src.postproc.utils import scrubbing, locate_electrodes
import pandas as pd

base_path = '/home/asier/git/ruber/data/processed/fmriprep/sub-001/func/'

confounds = opj(base_path, 'sub-001_task-rest_bold_confounds.tsv')
preproc_data =  opj(base_path,  'sub-001_task-rest_bold_space-MNI152NLin2009cAsym_preproc.nii.gz')
confounds = pd.read_csv(confounds, delimiter='\t', na_values='n/a').fillna(0)

# to be build in subject space. 
# TODO: Check if this atlas is the same for all subjects (it should)
atlas_2514 = '/home/asier/git/ruber/data/external/bha_atlas_2514_1mm_mni09c.nii.gz'
atlas_2514_img = nib.load(atlas_2514)
fmri = nib.load(preproc_data)
resampled_2514_atlas = resample_img(atlas_2514_img, target_affine=fmri.affine,
                                    interpolation='nearest')
nib.save(resampled_2514_atlas, opj(base_path, 'sub-001_atlas_2514_bold_space.nii.gz'))

# to be build in subject space
atlas_2754 = '/home/asier/git/ruber/data/external/bha_atlas_2754_1mm_mni09c.nii.gz'
atlas_2754_img = nib.load(atlas_2754)
fmri = nib.load(preproc_data)
resampled_2754_atlas = resample_img(atlas_2754_img, target_affine=fmri.affine,
                                    interpolation='nearest')
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

confounds_matrix = confounds[confounds_id].as_matrix()

# atlas_2514
masker = NiftiLabelsMasker(labels_img=atlas_2514, background_label=0,
                           verbose=5, detrend=True, standardize=True, t_r=2.72,
                           smoothing_fwhm=6, low_pass=0.1, high_pass=0.01)

time_series_2514 = masker.fit_transform(preproc_data,
                                        confounds=confounds_matrix)

# atlas_2754
masker = NiftiLabelsMasker(labels_img=atlas_2754, background_label=0,
                           verbose=5, detrend=True, standardize=True, t_r=2.72,
                           smoothing_fwhm=6, low_pass=0.1, high_pass=0.01)

time_series_2754 = masker.fit_transform(preproc_data,
                                        confounds=confounds_matrix) 

# 2.- Scrubbing
# extract FramewiseDisplacement
FD = confounds.iloc[:,5].as_matrix()
thres = 0.2

time_series_2514 = scrubbing(time_series_2514, FD, thres)
time_series_2754 = scrubbing(time_series_2754, FD, thres)

np.savetxt(opj(base_path, 'time_series_2514.txt'), time_series_2514)
np.savetxt(opj(base_path, 'time_series_2754.txt'), time_series_2754)



"""
Extract brain from electrodes T1W -> this to BIDS
"""

T1='/home/asier/Desktop/test_ruber/t1.nii.gz'
${FSLDIR}/bin/bet $T1 /home/asier/Desktop/test_ruber/T1_brain -B -f "0.1" -s -m 

"""
Atlas to subject space
"""
                         
flirt -in /home/asier/Desktop/test_ruber/T1_brain \
-ref /home/asier/git/ruber/data/external/standard_mni_asym_09c/mni_icbm152_t1_tal_nlin_asym_09c_brain.nii \
-cost mutualinfo -out /home/asier/Desktop/test_ruber/t1_brain_09c                    

# TODO: Include this in the pipeline in an organize way, preferrably following BIDS

elec_file = '/home/asier/Desktop/test_ruber/sub001elec.loc'

locate_electrodes(elec_file, 
                  atlas_2514, 
                  neighbours=0)

locate_electrodes(elec_file, 
                  atlas_2754, 
                  neighbours=0)
    
locate_electrodes(elec_file, 
                  atlas_2514, 
                  neighbours=1)
        
locate_electrodes(elec_file, 
                  atlas_2754, 
                  neighbours=1)
    

    
    
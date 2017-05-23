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
    

    
    
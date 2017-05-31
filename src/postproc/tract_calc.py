#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 29 09:41:47 2017

@author: asier
"""

from src.env import DATA
import os.path as op
from os.path import join as opj
from src.postproc.utils import execute
import numpy as np

ses = 'ses-presurg'
sub = 'sub-001'
output_name = sub

command = ['vtkstreamlines', '<',
           opj(DATA, 'processed', 'tract',
           '_session_id_' + ses + '_subject_id_' + sub, 'tracts.Bfloat_2514'),
           '>', output_name + '.vtk',
           ]

for output in execute(command):
    print(output)


vtkstreamlines < tracts.Bfloat_2514 > test.vtk

fslmaths eddy_corrected_denoised_DT_FA.nii -mul 0 -add 1 -roi 63 1 47 1 36 1 0 1 ACCpoint -odt float
fslmaths ACCpoint -kernel sphere 5 -fmean ACCsphere -odt float
fslmaths ACCsphere.nii.gz -bin ACCsphere_bin.nii.gz

## transform to dwi space! either with an omat from brain09c to dwi_brain or trying to find transform matrix 09c to dwi

cat tracts.Bfloat_2514 | procstreamlines -waypointfile ACCsphere_bin.nii.gz -endpointfile rsub-001_ses-presurg_atlas_2514.nii | conmat -targetfile rsub-001_ses-presurg_atlas_2514.nii 

cat tracts.Bfloat | counttracts
cat A_oneDT_1.Bfloat | procstreamlines -outputroot A_twoROI_ -outputtracts -waypointfile subA2ROI -regionindex 1
vtkstreamlines < streamlines.Bfloat > streamlines.vtk







struct_file = '/home/asier/Desktop/test_track/conmat_sc.csv'

struct_mat = np.loadtxt(struct_file, delimiter=',', skiprows=1)

np.count_nonzero(struct_mat)
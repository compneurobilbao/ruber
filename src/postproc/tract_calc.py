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
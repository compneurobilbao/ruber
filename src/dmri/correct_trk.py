#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  7 14:28:31 2017

@author: asier
"""

from nibabel.streamlines import save, Tractogram
from nibabel.streamlines.trk import TrkFile

a = TrkFile.load('/home/asier/git/ruber/data/processed/tract/_session_id_ses-presurg_subject_id_sub-001/tracts.trk')
save(Tractogram(a.streamlines, affine_to_rasmm=a.affine),'/home/asier/git/ruber/data/processed/tract/_session_id_ses-presurg_subject_id_sub-001/tracts_test.trk')

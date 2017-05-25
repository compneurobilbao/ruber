# -*- coding: utf-8 -*-

from src.postproc.utils import load_elec_file
from src.env import DATA, ATLAS_TYPES, CONFOUNDS_ID
import os.path as op
from os.path import join as opj
import numpy as np


SUBJECT_LIST = ['sub-001']
SESSION_LIST = ['ses-presurg']

if __name__ == "__main__":

    sub_ses_comb = [[subject, session] for subject in SUBJECT_LIST
                    for session in SESSION_LIST]

    for sub, ses in sub_ses_comb:
        for atlas in ATLAS_TYPES:
            # load ROI location of each contact
            elec_file = opj(DATA, 'raw', 'bids', sub, 'electrodes',
                            'sub-001_elec_2514_closest_rois.roi')
            elec_location_mni09 = load_elec_file(elec_file)

            # load struct
            
            # load function (conn matrix?)
            struct_file = opj(EXTERNAL, 'bha_' + atlas + '_1mm_mni09c.nii.gz')

# -*- coding: utf-8 -*-
from src.env import DATA

import os
from os.path import join as opj
import numpy as np
import scipy.io as sio

RAW_ELEC = opj(DATA, 'raw', 'elec_record')
PROCESSED_ELEC = opj(DATA, 'processed', 'elec_record')

SUBJECTS = ['sub-001', 'sub-002', 'sub-003', 'sub-004']
RITHMS = ['prefiltered', 'filtered', 'delta', 'theta', 'alpha', 'beta', 'gamma', 'gamma_high']


file = "/home/asier/git/ruber/data/raw/elec_record/sub-001/interictal/interictal_1_fb_5_numStd_2.mat"






def create_active_state_records():
    
    for sub in SUBJECTS:
        
        # create active state folder for sub
        sub_dir = opj(PROCESSED_ELEC, sub)
        if not os.path.exists(sub_dir):
            os.makedirs(sub_dir)
            
        for rit_num_code, rit in enumerate(RITHMS, 1):
            
            # create rithm folder in sub
            rithm_dir = opj(sub_dir, rit)
            if not os.path.exists(rithm_dir):
                os.makedirs(rithm_dir)
                
            for chunk in range (1, 13): # 12 chunks of 1 sec

                # load real data sub/rithm
                file = opj(PROCESSED_ELEC, sub, 'interictal_not_regressed',
                           rit, 'interictal_' + str(chunk) + '.npy')
                real_data = np.load(file)
                
                # num_elec
                
                # load AS data
                file = opj(RAW_ELEC, sub, 'interictal',
                           'interictal_' + str(chunk) + '_fb_' + 
                           str(rit_num_code) + '_numStd_2.mat')
                as_data = np.array(sio.loadmat(file).get('activeState'))

                
                active_state_to_signal()
                
                real_data_filtered_with_active_state()
                
                save
    
    return
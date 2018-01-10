# -*- coding: utf-8 -*-
from src.env import DATA

import os
from os.path import join as opj
import numpy as np
import scipy.io as sio

RAW_ELEC = opj(DATA, 'raw', 'elec_record')
PROCESSED_ELEC = opj(DATA, 'processed', 'elec_record')

SUBJECTS = ['sub-001', 'sub-002', 'sub-003', 'sub-004']
RITHMS = ['prefiltered', 'filtered', 'delta', 'theta',
          'alpha', 'beta', 'gamma', 'gamma_high']


def active_state_to_signal(as_data, real_data):
    """
    Takes real data matrix and active state data in Paolo's format.
    Creates a 0-1 matrix with Paolo's info and filters the real data with it.

    The output is the signal itself, with the non active state points removed
    """

    num_elec = as_data.shape[1]
    output_data = np.zeros((30000, num_elec))

    for elec_idx in range(num_elec):
        for pair in as_data[0][elec_idx]:
            output_data[pair[0]-1:pair[1], elec_idx] = 1

        output_data[:, elec_idx] = (output_data[:, elec_idx] *
                                    real_data[:, elec_idx])

    return output_data


def create_active_state_records():

    for sub in SUBJECTS:
        # create active state folder for sub
        sub_dir = opj(PROCESSED_ELEC, sub, 'active_state')
        if not os.path.exists(sub_dir):
            os.makedirs(sub_dir)

        for rit_num_code, rit in enumerate(RITHMS, 1):

            if rit == 'prefiltered':
                continue
            # create rithm folder in sub
            rithm_dir = opj(sub_dir, rit)
            if not os.path.exists(rithm_dir):
                os.makedirs(rithm_dir)

            for chunk in range(1, 13):  # 12 chunks of 1 sec

                # load real data sub/rithm
                file = opj(PROCESSED_ELEC, sub, 'interictal_not_regressed',
                           rit, 'interictal_' + str(chunk) + '.npy')
                real_data = np.load(file)

                # load AS data
                file = opj(RAW_ELEC, sub, 'interictal',
                           'interictal_' + str(chunk) + '_fb_' +
                           str(rit_num_code) + '_numStd_2.mat')
                as_data = np.array(sio.loadmat(file).get('activeState'))

                as_signal = active_state_to_signal(as_data, real_data)

                output_file = opj(rithm_dir,
                                  'active_state_' + str(chunk) + '.npy')
                np.save(output_file, as_signal)
    return

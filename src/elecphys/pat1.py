# -*- coding: utf-8 -*-
from src.env import DATA

import os
import os.path as op
from os.path import join as opj
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import remez, filtfilt

from src.elecphys.utils import (clean_file, 
                                clean_all_files_and_convert_to_npy,
                                bandpass_filter,
                                regress_signal,
                                filter_and_save,
                                )



def reorder_pat1_elec(elec_data):

    elec_data_ordered = np.zeros(elec_data.shape, dtype='float16')

    # OIM
    elec_data_ordered[:, 24:36] = elec_data[:, :12]
    # OIL
    elec_data_ordered[:, 12:24] = elec_data[:, 12:24]
    # OSM
    elec_data_ordered[:, 41:49] = elec_data[:, 24:32]
    # OSL
    elec_data_ordered[:, 36:41] = elec_data[:, 32:37]
    # TI
    elec_data_ordered[:, 49:57] = elec_data[:, 37:45]
    # A
    elec_data_ordered[:, :12] = elec_data[:, 45:57]

    return elec_data_ordered


def reorder_and_regress_pat1_elec(elec_data):

    elec_data_ordered = np.zeros(elec_data.shape, dtype='float16')

    # OIM
    elec_data_ordered[:, 24:36] = regress_signal(elec_data[:, :12])
    # OIL
    elec_data_ordered[:, 12:24] = regress_signal(elec_data[:, 12:24])
    # OSM
    elec_data_ordered[:, 41:49] = regress_signal(elec_data[:, 24:32])
    # OSL
    elec_data_ordered[:, 36:41] = regress_signal(elec_data[:, 32:37])
    # TI
    elec_data_ordered[:, 49:57] = regress_signal(elec_data[:, 37:45])
    # A
    elec_data_ordered[:, :12] = regress_signal(elec_data[:, 45:57])

    return elec_data_ordered


def create_pat1_files():

    
    input_path = '/home/asier/git/ruber/data/raw/elec_record/sub-001/interictal'
    out_path_reg = '/home/asier/git/ruber/data/processed/elec_record/sub-001/interictal_regressed'
    out_path_not_reg = '/home/asier/git/ruber/data/processed/elec_record/sub-001/interictal_not_regressed'

    
    for file in os.listdir(input_path):
        print(file)
        elec_data = np.load(opj(input_path, file))
        
        
        
        
        
        
        elec_data = reorder_and_regress_pat1_elec(elec_data)
    
        fs = 500
        lowcut = 0.05
        highcut = 70
        output = opj(output_path, 'filtered', file)
        filter_and_save(elec_data,  lowcut, highcut, fs, output)
    
        lowcut = 0.5
        highcut = 3
        output = opj(output_path, 'delta', file)
        filter_and_save(elec_data,  lowcut, highcut, fs, output)
    
        lowcut = 3
        highcut = 7
        output = opj(output_path, 'theta', file)
        filter_and_save(elec_data,  lowcut, highcut, fs, output)
    
        lowcut = 7
        highcut = 13
        output = opj(output_path, 'alpha', file)
        filter_and_save(elec_data,  lowcut, highcut, fs, output)
    
        lowcut = 13
        highcut = 30
        output = opj(output_path, 'beta', file)
        filter_and_save(elec_data,  lowcut, highcut, fs, output)
    
        lowcut = 30
        highcut = 70
        output = opj(output_path, 'gamma', file)
        filter_and_save(elec_data,  lowcut, highcut, fs, output)
    
        lowcut = 70
        highcut = 249
        output = opj(output_path, 'gamma_high', file)
        filter_and_save(elec_data,  lowcut, highcut, fs, output)


# -*- coding: utf-8 -*-
from src.env import DATA

import os
import os.path as op
from os.path import join as opj
import numpy as np
import tempfile
import shutil
import matplotlib.pyplot as plt
from scipy.signal import remez, filtfilt

from src.elecphys.utils import (clean_file, 
                                clean_all_files_and_convert_to_npy,
                                bandpass_filter,
                                )

"""
Code for extracting electrophysiological timeseries
"""

sub = 'sub-004'
INTERICTAL_DATA = opj(DATA, 'raw', 'elec_record', sub, 'chuncks')
clean_all_files_and_convert_to_npy(INTERICTAL_DATA)

chunck = str(2)
file = opj(INTERICTAL_DATA, 'chunck_' + chunck + '.npy')
contact_num = np.load(file)
plt.plot(contact_num)

interictal = str(5)
interictal_1 = contact_num[30000:60000, :]
np.save('/home/asier/git/ruber/data/raw/elec_record/'+sub+'/interictal/interictal_'+interictal+'.npy',interictal_1)

"""
Code to play with elec.phys timeseries
"""

contact_num = np.load(file[:-4] + '.npy')
fs = 500
lowcut = 0.05
highcut = 70

file = '/home/asier/git/ruber/data/raw/elec_record/sub-001/interictal/chunck_1.npy'
contact_num = np.load(file)


for i in range(57):
    plt.plot(contact_num[:, i])  # /max(abs(matrix(:,i)))+1*(i-1))
    
for i in [0, 3]:
    plt.plot(contact_num[:, i]-contact_num[:, i+1])


filtered = np.zeros((contact_num.shape))
for i in range(57):
    filtered[:, i] = bandpass_filter(contact_num[:, i], lowcut, highcut, fs)

for i in range(57):
    plt.plot(filtered[:, i])  # /max(abs(matrix(:,i)))+1*(i-1))
    


plt.plot(filtered[:, 32]-filtered[:, 33], linewidth=0.1)

plt.plot(filtered[:, 0][1000:]-filtered[:, 1][1000:])
plt.plot(filtered[:, 32][1000:]-filtered[:, 33][1000:], linewidth=0.1)

plt.plot(contact_num[:, 1]-contact_num[:, 2])
plt.plot(contact_num[:, 31]-contact_num[:, 32], linewidth=0.1)

plt.plot(filtered[:, 31][1000:])


"""
analysis august -> report
"""

### FILTER ###

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


def filter_and_save(elec_data, lowcut, highcut, fs, output_path):
    filtered = np.zeros((elec_data.shape))
    for i in range(57):
        filtered[:, i] = bandpass_filter(elec_data[:, i], lowcut, highcut, fs)
    np.save(output_path, filtered)


input_path = '/home/asier/git/ruber/data/raw/elec_record/sub-001/interictal'
output_path = '/home/asier/git/ruber/data/interim/elec_record/sub-001/interictal'

for file in os.listdir(input_path):
    print(file)
    elec_data = np.load(opj(input_path, file))
    elec_data = reorder_pat1_elec(elec_data)

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



### CORRELATION ###
input_path = '/home/asier/git/ruber/data/interim/elec_record/sub-001/interictal'
rithms = ['filtered', 'delta', 'theta', 'alpha', 'beta', 'gamma']

for rithm in rithms:
    
    all_conn_mat = np.zeros((12, 57, 57))
    
    for i, file  in enumerate(os.listdir(opj(input_path, rithm))):
        print(file)
        elec_data = np.load(opj(input_path, rithm, file))
        
        elec_conn_mat = np.zeros((57, 57))
        elec_conn_mat = np.corrcoef(elec_data.T)
        all_conn_mat[i, :, :] = elec_conn_mat

    con_mat = np.mean(all_conn_mat,0)

    plot_matrix(con_mat, elec_tags)

#    np.save(opj('/home/asier/git/ruber/reports/figures/sub-001',rithm),
#            con_mat)


#    for i in range(elec_data.shape[1]):
#        plt.plot(elec_data[:, i])

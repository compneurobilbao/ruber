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

INTERICTAL_DATA = opj(DATA, 'raw', 'elec_record', 'sub-001', 'interictal', 'exported')

clean_all_files_and_convert_to_npy(INTERICTAL_DATA)

test = '/media/asier/DISK_IMG/test.txt'
clean_file(test)

contact_num = np.loadtxt(test,
                         dtype='float32',
                         delimiter='\t',
                         usecols=range(3, 60))


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



INTERICTAL_DATA = opj(DATA, 'raw', 'elec_record', 'sub-001', 'interictal', 'exported')
clean_all_files_and_convert_to_npy(INTERICTAL_DATA)

file = '/home/asier/git/ruber/data/raw/elec_record/sub-001/interictal/chunck_3.npy'
contact_num = np.load(file)
plt.plot(contact_num)

interictal_12 = contact_num[1120000:1240000, :]
np.save('/home/asier/git/ruber/data/raw/elec_record/sub-001/interictal/interictal_12.npy',interictal_12)


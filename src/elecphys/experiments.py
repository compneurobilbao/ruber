# -*- coding: utf-8 -*-
from src.env import DATA

import os
import os.path as op
from os.path import join as opj
import numpy as np
import tempfile
import shutil
import matplotlib.pyplot as plt


INTERICTAL_DATA = opj(DATA, 'raw', 'elec_record', 'sub-001', 'interictal')


def clean_file(file_path):

    wrong_word = 'BREAK'
    temp_file = tempfile.mkstemp()[1]

    with open(file_path) as oldfile, open(temp_file, 'w') as newfile:
        for line in oldfile:
            if wrong_word not in line:
                newfile.write(line)

    shutil.move(temp_file, file_path)


def clean_all_files_and_convert_to_npy():

    for filename in os.listdir(INTERICTAL_DATA):
        file = opj(INTERICTAL_DATA, filename)
        if filename.endswith(".txt"):
            clean_file(file)
            with open(file, 'r') as f:
                ncols = len(f.readline().split('\t'))

            numpy_matrix = np.loadtxt(file,
                                      dtype='float32',
                                      delimiter='\t',
                                      usecols=range(3, ncols-1))
            np.save(file[:-4], numpy_matrix)

from scipy.signal import butter, lfilter



from scipy.signal import butter, lfilter

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq

    b, a = butter(order, [low, high], btype='band')
    y = lfilter(b, a, data)
    return y




test = '/media/asier/DISK_IMG/test.txt'
clean_file(test)

contact_num = np.loadtxt(test,
                         dtype='float32',
                         delimiter='\t',
                         usecols=range(3, 60))

fs = 2000
lowcut = 0.5
highcut = 70



for i in range(57):
    plt.plot(contact_num[:, i])  # /max(abs(matrix(:,i)))+1*(i-1))
    
for i in [0, 3]:
    plt.plot(contact_num[:, i]-contact_num[:, i+1])


filtered = np.zeros((contact_num.shape))
for i in range(57):
    filtered[:, i] = butter_bandpass_filter(contact_num[:, i], lowcut, highcut, fs, order=6)



plt.plot(filtered[:, 0]-filtered[:, 1])
plt.plot(contact_num[:, 1]-contact_num[:, 2])
plt.plot(contact_num[:, 0]-contact_num[:, 1])


plt.plot(contact_num[:, 1])

signal = contact_num[:, 1]-contact_num[:, 2]

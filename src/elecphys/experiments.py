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


from scipy.signal import butter, lfilter, remez, filtfilt


def bandpass_filter(data, fs, lowcut, highcut):

    order = 1000
    ns, num_channels = np.shape(data)
    if ns < 3 * order:
        order = np.floor(ns/3)

    Fstop1 = lowcut - 0.001
    Fpass1 = lowcut

    Fpass2 = highcut
    Fstop2 = highcut + 0.005
    Wstop1 = 10
    Wpass = 1
    Wstop2 = 10
    dens = 20

    b = remez(order+1, 
              bands=np.array([0, Fstop1, Fpass1, Fpass2, Fstop2, fs/2]),
              desired=[0, 1, 0],
              Hz=fs,
              weight=[Wstop1, Wpass, Wstop2],
              grid_density=dens)
    
    y = filtfilt(b, 1, data[:, 0])
    return y


plt.plot(data[:, 0])
plt.figure()
plt.plot(y)

test = '/media/asier/DISK_IMG/test.txt'
clean_file(test)

contact_num = np.loadtxt(test,
                         dtype='float32',
                         delimiter='\t',
                         usecols=range(3, 60))


contac_num = np.load(file[:-4] + '.npy')
fs = 500
lowcut = 0.05
highcut = 70



for i in range(57):
    plt.plot(contact_num[:, i])  # /max(abs(matrix(:,i)))+1*(i-1))
    
for i in [0, 3]:
    plt.plot(contact_num[:, i]-contact_num[:, i+1])


filtered = np.zeros((contact_num.shape))
for i in range(57):
    filtered[:, i] = butter_bandpass_filter(contact_num[:, i], lowcut, highcut, fs, order=1)



plt.plot(filtered[:, 0][1000:]-filtered[:, 1][1000:])
plt.plot(filtered[:, 32][1000:]-filtered[:, 33][1000:], linewidth=0.1)

plt.plot(contact_num[:, 1]-contact_num[:, 2])
plt.plot(contact_num[:, 31]-contact_num[:, 32], linewidth=0.1)

plt.plot(filtered[:, 31][1000:])
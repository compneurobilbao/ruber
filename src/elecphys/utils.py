# -*- coding: utf-8 -*-
"""
Utilities to help in the deep electrode signals pre-processing
"""
import os
from os.path import join as opj
import numpy as np
import tempfile
import shutil
from scipy.signal import remez, filtfilt


def clean_file(file_path):

    wrong_word = 'BREAK'
    temp_file = tempfile.mkstemp()[1]

    with open(file_path, encoding='utf-16') as oldfile, \
            open(temp_file, 'w') as newfile:
        for line in oldfile:
            if wrong_word not in line:
                newfile.write(line)

    shutil.move(temp_file, file_path)


def clean_all_files_and_convert_to_npy(data_path):

    for filename in os.listdir(data_path):
        file = opj(data_path, filename)
        if filename.endswith(".txt"):
            try:
                clean_file(file)
            except:
                pass
            with open(file, 'r') as f:
                ncols = len(f.readline().split('\t'))

            numpy_matrix = np.loadtxt(file,
                                      dtype='float32',
                                      delimiter='\t',
                                      usecols=range(3, ncols-1))
            np.save(file[:-4], numpy_matrix)


def bandpass_filter(data, lowcut, highcut, fs):

    order = 1000
    ns = np.shape(data)[0]
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

    y = filtfilt(b, 1, data)
    return y


def filter_and_save(elec_data, lowcut, highcut, fs, output_path):
    import scipy.io as sio

    rows, cols = elec_data.shape
    filtered = np.zeros((rows, cols))
    for i in range(cols):
        filtered[:, i] = bandpass_filter(elec_data[:, i], lowcut, highcut, fs)
    np.save(output_path, filtered)
    sio.savemat(output_path[:-4] + '.mat', {'data': filtered})


def regress_signal(elec_data):
    """
    the "electrodeSignal" must be a matrix for a given electrode (OIL, OIM etc)
    If the electrode has N recording sites (cols), and the samples in the chunk
    signal are M (rows), the variable  "electrodeSignal" is a matrix of MxN
    """

    regressed = np.zeros((elec_data.shape))
    for i in range(elec_data.shape[1]):
        xx = np.column_stack((np.ones(elec_data.shape[0]),
                              np.mean(np.delete(elec_data, i, axis=1), 1)))
        wml = np.dot(np.dot(np.linalg.pinv(np.dot(xx.T,
                                                  xx)),
                            xx.T),
                     elec_data[:, i])
        regressed[:, i] = elec_data[:, i] - wml[0] - wml[1] * \
            np.mean(np.delete(elec_data, i, axis=1), 1)

    return regressed


def filter_and_save_all_bands(output_path, elec_data, file):
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

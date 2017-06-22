# -*- coding: utf-8 -*-
from src.env import DATA

import os
import os.path as op
from os.path import join as opj
import numpy as np
import tempfile
import shutil

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


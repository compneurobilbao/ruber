#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 31 10:55:25 2017

@author: asier
"""

import numpy as np
import scipy.io as sio
import os

root = '/home/asier/git/ruber/data/raw/elec_record'


for path, subdirs, files in os.walk(root):
    for name in files:
        file_path = os.path.join(path, name)
        if file_path.endswith('.npy'):
            data = np.load(file_path)
            sio.savemat(file_path[:-4] + '.mat', {'data': data})
            
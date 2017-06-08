# -*- coding: utf-8 -*-

from src.postproc.utils import load_elec_file, order_dict
from src.env import DATA, ATLAS_TYPES
#import os.path as op
from os.path import join as opj
import numpy as np
from matplotlib import pyplot as plt
from nilearn.connectome import ConnectivityMeasure


SUBJECT_LIST = ['sub-001']
SESSION_LIST = ['ses-presurg']


def plot_matrix(matrix, elec_tags):
    plt.figure(figsize=(10, 10))
    # Mask the main diagonal for visualization:

    plt.imshow(matrix, interpolation="nearest", cmap="RdBu_r")
    # vmax=0.8, vmin=-0.8)

    # Add labels and adjust margins
    plt.xticks(range(len(elec_tags)), elec_tags, rotation=90)
    plt.yticks(range(len(elec_tags)), elec_tags)
    plt.gca().yaxis.tick_right()
    plt.subplots_adjust(left=.01, bottom=.3, top=.99, right=.62)


if __name__ == "__main__":

    sub_ses_comb = [[subject, session] for subject in SUBJECT_LIST
                    for session in SESSION_LIST]

    for sub, ses in sub_ses_comb:
        for atlas in ATLAS_TYPES:
            # FUNCTION MATRIX
            elec_file = opj(DATA, 'raw', 'bids', sub, 'electrodes',
                            'sub-001_' + atlas + '_closest_rois.roi')
            elec_location_mni09 = load_elec_file(elec_file)

            ordered_elec = order_dict(elec_location_mni09)

            elec_tags = list(ordered_elec.keys())
            elec_rois = np.array(list(ordered_elec.values()))[:, 0]

            idx = np.ix_(elec_rois, elec_rois)

            # load function (conn matrix?)
            func_file = opj(DATA, 'processed', 'fmriprep', sub, ses, 'func',
                            'time_series_' + atlas + '.txt')

            func_mat = np.loadtxt(func_file)

            correlation_measure = ConnectivityMeasure(kind='correlation')
            corr_mat = correlation_measure.fit_transform([func_mat])[0]
            plot_matrix(corr_mat[idx], elec_tags)

            # STRUCT MATRIX
            struct_mat = np.load(opj(DATA, 'raw', 'bids', sub, 'electrodes',
                                     ses, 'con_mat_' + atlas + '.npy'))

            plot_matrix(struct_mat, elec_tags)


sc_camino = '/home/asier/git/ruber/data/processed/tract/_session_id_ses-presurg_subject_id_sub-001/conmat_atlas_2514_sc.csv'
sc_matlab = '/home/asier/Desktop/test_track/fiber_number.txt'

camino = np.loadtxt(sc_camino, delimiter=',', skiprows=1)
matlab = np.loadtxt(sc_matlab)


"""
test area
"""

import h5py
import numpy as np

file1 = '/home/asier/git/ruber/data/raw/elec_record/sub-001/rec_2_export_all_selectedChannels_1_53_63_64.mat'
f = h5py.File(file1)
for k, v in f.items():
    elec1 = np.array(v, dtype='float16')

file2 = '/home/asier/git/ruber/data/raw/elec_record/sub-001/rec_2_export_all_selectedChannels_54_56.mat'
f = h5py.File(file2)
for k, v in f.items():
    elec2 = np.array(v, dtype='float16')

elec_data = np.concatenate((elec1[:-2], elec2)) # from elec1 2 last electrodes are EKG
# np.save('/home/asier/git/ruber/data/raw/elec_record/sub-001/elec_data',
#         elec_data)

elec_conn_mat = np.zeros((57, 57))
elec_conn_mat[:56,:56] = np.corrcoef(elec_data[:,:25611761])

np.save('/home/asier/git/ruber/data/raw/elec_record/sub-001/elec_con_mat',
         elec_conn_mat)

elec_data.shape
elec_conn_mat.shape

# Reorder elec tags
sc_mat = np.zeros((57, 57))
el_tags = []
# OIM
el_tags[:12] = elec_tags[24:36]
sc_mat[24:36, 24:36] = struct_mat[24:36, 24:36]
# OIL
el_tags.extend(elec_tags[12:24])
sc_mat[12:24, 12:24] = struct_mat[12:24, 12:24]
# OSM 
el_tags.extend(elec_tags[41:49])
sc_mat[41:49, 41:49] = struct_mat[41:49, 41:49]
# OSL
el_tags.extend(elec_tags[36:41])
sc_mat[36:41, 36:41] = struct_mat[36:41, 36:41]
# TI
el_tags.extend(elec_tags[49:57])
sc_mat[49:57, 49:57] = struct_mat[49:57, 49:57]
# A
el_tags.extend(elec_tags[:12])
sc_mat[:12, :12] = struct_mat[:12, :12]


plot_matrix(sc_mat, el_tags)
plot_matrix(elec_conn_mat, el_tags)



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


def log_transform(im):
    '''returns log(image) scaled to the interval [0,1]'''
    try:
        (min, max) = (im[im > 0].min(), im.max())
        if (max > min) and (max > 0):
            return (np.log(im.clip(min, max)) - np.log(min)) / (np.log(max) - np.log(min))
    except:
        pass
    return im


def plot_matrix(matrix, elec_tags, log=False):
    plt.figure(figsize=(10, 10))
    # Mask the main diagonal for visualization:

    if log:
        matrix = log_transform(matrix)

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
        # FUNCTION MATRIX
        elec_file = opj(DATA, 'raw', 'bids', sub, 'electrodes',
                        'elec.loc')
        elec_location_mni09 = load_elec_file(elec_file)

        ordered_elec = order_dict(elec_location_mni09)

        elec_tags = list(ordered_elec.keys())

        # load function (conn matrix?)
        func_file = opj(DATA, 'processed', 'fmriprep', sub, ses, 'func',
                        'time_series_noatlas.txt')

        func_mat = np.loadtxt(func_file)

        correlation_measure = ConnectivityMeasure(kind='correlation')
        corr_mat = correlation_measure.fit_transform([func_mat])[0]

        # STRUCT MATRIX
        struct_mat = np.load(opj(DATA, 'raw', 'bids', sub, 'electrodes',
                                 ses, 'con_mat_noatlas.npy'))

        
        
        elec_mat = np.load('/home/asier/git/ruber/data/raw/elec_record/sub-001/elec_con_mat.npy')
        
        plot_matrix(corr_mat, elec_tags)
        plt.title('fMRI connectivity matrix')
        plt.savefig(opj('/home/asier/git/ruber/reports/figures/', sub, 'fMRI_conmat'))

        plot_matrix(struct_mat, elec_tags, log=True)
        plt.title('DWI connectivity matrix')
        plt.savefig(opj('/home/asier/git/ruber/reports/figures/', sub, 'DWI_conmat'))

        plot_matrix(elec_mat, elec_tags)
        plt.title('deep electrode connectivity matrix')
        plt.savefig(opj('/home/asier/git/ruber/reports/figures/', sub, 'deep_conmat'))

        plt.figure()
        plt.scatter(log_transform(struct_mat), corr_mat)
        plt.title('#Streamlines vs fMRI corr values')
        plt.xlabel('log(#streamlines)')
        plt.ylabel('correlation values')
        plt.savefig(opj('/home/asier/git/ruber/reports/figures/', sub, 'scatter_DWI_fMRI'))
        plt.close("all")
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
elec_data_ordered = np.zeros((57, elec_data.shape[1]), dtype='float16')

# OIM
elec_data_ordered[24:36, :] = elec_data[:12, :]
# OIL
elec_data_ordered[12:24, :] = elec_data[12:24, :]
# OSM
elec_data_ordered[41:49, :] = elec_data[24:32, :]
# OSL
elec_data_ordered[36:41, :] = elec_data[32:37, :]
# TI
elec_data_ordered[49:57, :] = elec_data[37:45, :]
# A
elec_data_ordered[:11, :] = elec_data[45:57, :]


elec_conn_mat = np.zeros((57, 57))
elec_conn_mat = np.corrcoef(elec_data_ordered[:, 13352000:13367000])

np.save('/home/asier/git/ruber/data/raw/elec_record/sub-001/elec_con_mat',
        elec_conn_mat)

plot_matrix(struct_mat, elec_tags, log=True)
plot_matrix(elec_conn_mat, elec_tags)

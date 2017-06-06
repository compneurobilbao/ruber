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


def plot_matrix(matrix, idx, elec_tags):
    plt.figure(figsize=(10, 10))
    # Mask the main diagonal for visualization:

    plt.imshow(matrix[idx], interpolation="nearest", cmap="RdBu_r")
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
            # load ROI location of each contact
            elec_file = opj(DATA, 'raw', 'bids', sub, 'electrodes',
                            'sub-001_' + atlas + '_1_neighbours.roi')
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

            # load struct
#            struct_file = opj(DATA, 'processed', 'tract', '_session_id_' +
#                              ses + '_subject_id_' + sub,
#                              'conmat_' + atlas + '_sc.csv')
#
            struct_mat = np.load(opj(DATA, 'raw', 'bids', sub, 'electrodes',
                                     ses, 'con_mat_' + atlas + '.npy'))

            plot_matrix(corr_mat, idx, elec_tags)
            plot_matrix(struct_mat, idx, elec_tags)

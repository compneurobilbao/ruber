# -*- coding: utf-8 -*-

from src.postproc.utils import load_elec_file, order_dict
from src.env import DATA
import os
from os.path import join as opj
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from nilearn.connectome import ConnectivityMeasure
from itertools import product

SUBJECT_LIST = ['sub-001', 'sub-002', 'sub-003', 'sub-004']
SESSION_LIST = ['ses-presurg']

CWD = os.getcwd()


def log_transform(im):
    '''returns log(image) scaled to the interval [0,1]'''
    try:
        (min, max) = (im[im > 0].min(), im.max())
        if (max > min) and (max > 0):
            return ((np.log(im.clip(min, max)) - np.log(min)) /
                    (np.log(max) - np.log(min)))
    except:
        pass
    return im


def multipage(filename, figs=None, dpi=200):
    pp = PdfPages(filename)
    if figs is None:
        figs = [plt.figure(n) for n in plt.get_fignums()]
    for fig in figs:
        fig.savefig(pp, format='pdf')
    pp.close()


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
    
    SPHERE_SIZE = [3]
    DENOISE_TYPE = ['gsr']

    for sub, ses in sub_ses_comb:
        for sphere, denoise_type in product(SPHERE_SIZE, DENOISE_TYPE):
        
            output_dir_path = opj(CWD, 'reports', 'figures', sub)
            if not os.path.exists(output_dir_path):
                os.makedirs(output_dir_path)
            # FUNCTION MATRIX
            elec_file = opj(DATA, 'raw', 'bids', sub, 'electrodes',
                            'elec.loc')
            elec_location_mni09 = load_elec_file(elec_file)
    
            ordered_elec = order_dict(elec_location_mni09)
    
            elec_tags = list(ordered_elec.keys())
    
            # load function (conn matrix?)
            func_file = opj(DATA, 'processed', 'fmriprep', sub, ses, 'func',
                            'time_series_noatlas_' + denoise_type + '_' +
                            str(sphere) + '.txt')
            func_mat = np.loadtxt(func_file)

            correlation_measure = ConnectivityMeasure(kind='correlation')
            corr_mat = correlation_measure.fit_transform([func_mat])[0]

            # STRUCT MATRIX
            struct_mat = np.load(opj(DATA, 'raw', 'bids', sub, 'electrodes', ses,
                                     'con_mat_noatlas_' +
                                     str(sphere) + '.npy'))

            plot_matrix(corr_mat, elec_tags)
            ax1 = plt.title('fMRI connectivity matrix: ' + denoise_type + ':' +
                            'sphere size: ' + str(sphere))
            fig1 = ax1.get_figure()

            plot_matrix(struct_mat, elec_tags, log=True)
            ax2 = plt.title('DWI connectivity matrix: ' +
                            'sphere size: ' + str(sphere))
            fig2 = ax2.get_figure()
            plt.close("all")
            plt.scatter(log_transform(struct_mat), corr_mat)
            ax3 = plt.title('#Streamlines vs fMRI corr values' + denoise_type +
                            ':' + 'sphere size: ' + str(sphere))
            plt.xlabel('log(#streamlines)')
            plt.ylabel('correlation values')
            fig3 = ax3.get_figure()

            multipage(opj(output_dir_path,
                          'scatter_DWI_fMRI_' + denoise_type + '_' +
                          str(sphere) + '.pdf'),
                      [fig1, fig2, fig3],
                      dpi=250)

            plt.close("all")





"""
analysis august -> report
"""


### CORRELATION ###
input_path = '/home/asier/git/ruber/data/interim/elec_record/sub-001/interictal_not_regressed'
rithms = ['filtered', 'delta', 'theta', 'alpha', 'beta', 'gamma', 'gamma_high']

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


    for i in range(elec_data.shape[1]):
        plt.plot(elec_data[:, i])
    for i in range(regressed.shape[1]):
        plt.plot(regressed[:, i])

# -*- coding: utf-8 -*-
from src.postproc.utils import load_elec_file, order_dict
from src.env import DATA
import os
from os.path import join as opj
import numpy as np
from matplotlib import pyplot as plt


from nilearn.connectome import ConnectivityMeasure
from itertools import product

CWD = os.getcwd()


def calculate_energy(elec_data, window_size=500):
    '''
    Calculates energy of a signal based on a sliding window
    '''
    power_data = elec_data * elec_data
    window = np.ones((window_size,))
    points, channels = power_data.shape

    energy = np.empty(((max(points, window_size) -
                        min(points, window_size) + 1),
                       channels))

    for i in range(channels):
        energy[:, i] = np.convolve(power_data[:, i], window, 'valid')

    return energy


def count_energy_over_percentile(energy, perc=95):
    '''
    For each channel, counts how many values of
    the energy go over the percentile value
    '''
    _, channels = energy.shape
    counter = np.empty((channels,))

    for i in range(channels):
        # Counts how many values of the energy go over the percentile value
        counter[i] = np.where(energy[:, i] > np.percentile(energy[:, i],
                                                           perc))[0].shape[0]

    return counter


#matplotlib.pyplot.hist(energy[:,i], bins=100)
#
#a = sc.signal.hilbert(elec_data[:, i])
#matplotlib.pyplot.plot(np.abs(a))
#matplotlib.pyplot.hist(np.log(np.abs(a)), bins=100)
#matplotlib.pyplot.hist(np.abs(a), bins=100)
#
#
#matplotlib.pyplot.plot(elec_data[:, i])

def figures_1():

    rithms = ['filtered', 'delta', 'theta', 'alpha', 'beta', 'gamma', 'gamma_high']
    SUBJECT_LIST = ['sub-001', 'sub-002', 'sub-003', 'sub-004']
    SESSION_LIST = ['ses-presurg']

    sub_ses_comb = [[subject, session] for subject in SUBJECT_LIST
                    for session in SESSION_LIST]

    SPHERE_SIZE = [3]
    DENOISE_TYPE = ['gsr']

    for sub, ses in sub_ses_comb:

        # fMRI and DTI
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

        # Electrophysiology
        for elec_reg_type in ['regressed', 'not_regressed']:
            input_path = opj(CWD, 'data', 'processed', 'elec_record',
                             sub, 'interictal_' + elec_reg_type)
            figures = []
            for rithm in rithms:

                # load random file
                random_data = np.load(opj(input_path, 'alpha',
                                          'interictal_1.npy'))
                contact_num = random_data.shape[1]
                all_conn_mat = np.zeros((12, contact_num, contact_num))

                files = [file for file in os.listdir(opj(input_path, rithm))
                         if file.endswith('npy')]
                for i, file  in enumerate(files):
                    elec_data = np.load(opj(input_path, rithm, file))
                    
                    elec_conn_mat = np.zeros((contact_num, contact_num))
                    elec_conn_mat = np.corrcoef(elec_data.T)
                    all_conn_mat[i, :, :] = elec_conn_mat
            
                con_mat = np.mean(all_conn_mat,0)
            
                plot_matrix(con_mat, elec_tags)
                ax = plt.title('Electrophysiology ' + elec_reg_type + ':' +
                                'rithm: ' + rithm)
                fig = ax.get_figure()
                figures.append(fig)
                plt.close()

            multipage(opj(output_dir_path,
                          'Electrophysiology_' + elec_reg_type +
                          '_bands.pdf'),
                          figures,
                          dpi=250)


def figures_2():

    rithms = ['filtered', 'delta', 'theta', 'alpha', 'beta', 'gamma', 'gamma_high']
    SUBJECT_LIST = ['sub-001', 'sub-002', 'sub-003', 'sub-004']
    SESSION_LIST = ['ses-presurg']

    sub_ses_comb = [[subject, session] for subject in SUBJECT_LIST
                    for session in SESSION_LIST]

    sphere = 3
    denoise_type = 'gsr'

    for i, sub_ses in enumerate(sub_ses_comb):
        sub, ses = sub_ses
        output_dir_path = opj(CWD, 'reports', 'figures', sub)
        if not os.path.exists(output_dir_path):
            os.makedirs(output_dir_path)
        # FUNCTION MATRIX
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

        th = 0
        struct_mat = log_transform(struct_mat)
        struct_mat_treatment = 'log_th_' + str(th)

        idx = np.where(struct_mat >= th)
        struct_mat = struct_mat[idx]
        corr_mat = corr_mat[idx]  # CAREFULL!! THIS IS WRONG IF TH>0, deletes FUNCTIONAL nodes

        # Electrophysiology
        for elec_reg_type in ['regressed', 'not_regressed']:
            input_path = opj(CWD, 'data', 'processed', 'elec_record',
                             sub, 'interictal_' + elec_reg_type)
            figures = []
            corr_values_struct = []
            corr_values_func = []

            random_data = np.load(opj(input_path, 'alpha',
                                      'interictal_1.npy'))
            contact_num = random_data.shape[1]
            all_conn_mat = np.zeros((12, contact_num, contact_num))
            for rithm in rithms:
                # load random file
                files = [file for file in os.listdir(opj(input_path, rithm))
                         if file.endswith('npy')]
                for i, file in enumerate(files):
                    elec_data = np.load(opj(input_path, rithm, file))

                    elec_conn_mat = np.zeros((contact_num, contact_num))
                    elec_conn_mat = np.corrcoef(elec_data.T)
                    all_conn_mat[i, :, :] = elec_conn_mat
                # Get elec con_mat
                con_mat = np.mean(all_conn_mat, 0)
                con_mat = con_mat[idx]

                # scatter vs struct
                corr_value = np.corrcoef(struct_mat,
                                         con_mat)[0][1]
                corr_values_struct.append(corr_value)

                plt.scatter(struct_mat, con_mat)
                ax = plt.title('R = ' + str(corr_value) +
                               '#Streamlines vs corr values of ' + rithm +
                               ' ' + elec_reg_type)
                plt.xlabel(struct_mat_treatment)
                plt.ylabel('elec corr')
                fig = ax.get_figure()
                figures.append(fig)
                plt.close()

                # scatter vs func
                corr_value = np.corrcoef(np.ndarray.flatten(struct_mat),
                                         np.ndarray.flatten(con_mat))[0][1]
                corr_values_func.append(corr_value)

                plt.scatter(corr_mat, con_mat)
                ax = plt.title('R = ' + str(corr_value) +
                               'Func corr vs corr values of ' + rithm +
                               ' ' + elec_reg_type)
                plt.xlabel('Func corr')
                plt.ylabel('elec corr')
                fig = ax.get_figure()
                figures.append(fig)
                plt.close()

            multipage(opj(output_dir_path,
                          'Scatter_func_struct_' + sub + '_' +
                          struct_mat_treatment + '_' +
                          elec_reg_type + ' ' +
                          '_bands.pdf'),
                      figures,
                      dpi=250)
            plt.close("all")

            np.save(opj(CWD, 'reports', 'stats',
                        'stats_' + sub + '_' +
                        struct_mat_treatment + '_' +
                        elec_reg_type),
                    np.array(corr_values_struct))

            np.save(opj(CWD, 'reports', 'stats',
                        'stats_' + sub + '_func_' +
                        elec_reg_type),
                    np.array(corr_values_func))
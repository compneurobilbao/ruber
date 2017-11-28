# -*- coding: utf-8 -*-
from src.postproc.utils import load_elec_file, order_dict
from src.env import DATA
import os
from os.path import join as opj
import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit

from nilearn.connectome import ConnectivityMeasure
from itertools import product

CWD = os.getcwd()


def calculate_energy(data, window_size=500):
    '''
    Calculates energy of a signal based on a sliding window
    '''
    power_data = data * data
    window = np.ones((window_size,))

    if data.ndim == 1:
        energy = np.convolve(power_data, window, 'valid')
    else:
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


def gauss(x, *p):
    A, mu, sigma = p
    return A*np.exp(-(x-mu)**2/(2.*sigma**2))


def calc_gaussian_fit(signal):
    # do gaussian fit; decide in next line if in log10 space or not
    # Paolo and stackoverflow.com/questions/11507028/fit-a-gaussian-function

    y, X = np.histogram(signal, density=True)

    s = max(signal) - min(signal)
    bins = np.arange(np.min(signal), np.max(signal), s/500)
    y, X = np.histogram(signal, bins)

    X_ = X[np.where((y > max(y) * 0.005))]
    y_ = y[np.where((y > max(y) * 0.005))]

    # p0 is the initial guess for the fitting coefficients (A, mu and sigma)
    p0 = [np.max(y_), np.mean(signal), 2*(np.std(signal))]

    coeff, var_matrix = curve_fit(gauss, X_, y_, p0=p0)

    mean_gauss_fit = coeff[1]
    std_gauss_fit = coeff[2]

    return mean_gauss_fit, std_gauss_fit


def calc_envelope_oscillations(signal, window_size=50, times_cyc_window=4):

    positive_part = signal
    positive_part[np.where(positive_part < 0)] = 0

    window = np.ones((window_size*times_cyc_window,))

    if signal.ndim == 1:
        envelope_oscillations = np.convolve(positive_part, window, 'valid')
    else:
        points, channels = signal.shape

        envelope_oscillations = np.empty(((max(points, window_size) -
                                           min(points, window_size) + 1),
                                         channels))

        for i in range(channels):
            envelope_oscillations[:, i] = np.convolve(positive_part[:, i],
                                                      window,
                                                      'valid')

    return envelope_oscillations/(window_size*times_cyc_window)


def calculate_active_state(signal, lower_band, sampling_freq=500,
                           cycles=2, times_cyc_window=4, apply_log=True,
                           std_parameter=3):
    """
    Calculates active state of a signal taking cycles into account.
    Input: 1D dimensional signal
    Output: Signal_response
    """

    window_size = np.int(np.ceil(cycles * sampling_freq / lower_band))
    if window_size > 50:
        window_size = 50
    energy = calculate_energy(signal, window_size)

    if apply_log is True:
        energy = np.log10(energy)

    mean_gauss_fit, std_gauss_fit = calc_gaussian_fit(energy)
    norm_sig = energy - mean_gauss_fit

    envelope_oscillations = calc_envelope_oscillations(norm_sig,
                                                       window_size,
                                                       times_cyc_window)

    # Find start-end peaks
    # parameter refractory time between peaks
    refract_time = cycles * 2
    # parameter number standard deviation above noise
    points = np.where(norm_sig > (std_parameter * std_gauss_fit))[0]
    refract_points = np.where(np.diff(points) > refract_time)[0]
    start_peaks = np.array((points[0], points[refract_points + 1]))
    # end_peaks = np.array((points[refract_points], points[-1]))

    # Find start end of envelope oscillation
    # parameter refractory time between env.oscs.
    refract_time = cycles
    points = np.where(envelope_oscillations > 0)[0]
    refract_points = np.where(np.diff(points) > refract_time)[0]
    start_env = np.concatenate(([points[0]], points[refract_points+1]))
    end_env = np.concatenate((points[refract_points], [points[-1]]))

    start_osc = []
    end_osc = []
    for i, peak in enumerate(start_peaks):
        for j, env in enumerate(start_env):
            if peak in np.arange(env, end_env[j]):
                start_osc.append(start_env[i])
                end_osc.append(end_env[j])

    active_state = np.zeros(envelope_oscillations.shape)
    for i, osc in enumerate(start_osc):
        active_state[osc:end_osc[i]] = 1

    return energy, active_state


def calc_active_state_elec(signals, lower_band):

    active_state = np.zeros((signals.shape))
    energy = np.zeros((signals.shape))

    for i in range(signals.shape[0]):
        signal = signals[i, :]
        energy[i, :], active_state[i, :] = calculate_active_state(signal,
                                                                  lower_band)

    return energy. active_state


def plot_active_state(signal, active_state, labels=[]):
    """
    Function to plot signals and their response in time.
    The aim is to see propagation of activations between electrodes.
    Does not work(yet) for 1D arrays
    """
    import matplotlib.collections as col

    if signal.ndim == 1:
        channels = 1
        height = 0.9
    else:
        points, channels = signal.shape
        height = 0.9/channels

    fig = plt.figure()

    yprops = dict(rotation=0,
                  horizontalalignment='right',
                  verticalalignment='center',
                  x=-0.01)

    axprops = dict(yticks=[])

    for i in range(channels):

        sig = signal[:, i]
        act_state = active_state[:, i]
        # [left, bottom, width, height] bottom and height are parameters!!
        ax = fig.add_axes([0.1, height * i+0.05, 0.8, height], **axprops)
        ax.plot(sig)

        collection = col.BrokenBarHCollection.span_where(x=range(len(sig)),
                                                         ymin=min(sig),
                                                         ymax=max(sig),
                                                         where=act_state > 0,
                                                         facecolor='green')

        ax.add_collection(collection)
        if labels:
            ax.set_ylabel(labels[i], **yprops)
        if i == 0:
            axprops['sharex'] = ax
            axprops['sharey'] = ax
        else:
            plt.setp(ax.get_xticklabels(), visible=False)


def calc_active_state_fmri(signal,
                           std_parameter=2):
    """
    Calculates active state of a signal taking positive values over a th
    into account.
    Input: Multidimensional signal
    Output: Signal_response
    """
    active_state_fmri = np.zeros((signal.shape()))

    std = np.std(signal[:])
    active_state_fmri[np.where(signal > 2*std)] = 1

    return active_state_fmri


# Calc active state for each electrode
# plot and save
# compare with MD outcomes
            
# do the same for fMRI
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

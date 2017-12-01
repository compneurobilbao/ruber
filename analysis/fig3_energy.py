# -*- coding: utf-8 -*-
from src.postproc.utils import load_elec_file, order_dict
from src.env import DATA
import os
from os.path import join as opj
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy.optimize import curve_fit
from collections import OrderedDict

from nilearn.connectome import ConnectivityMeasure
from itertools import product


CWD = os.getcwd()


def multipage(filename, figs=None, dpi=200):
    pp = PdfPages(filename)
    if figs is None:
        figs = [plt.figure(n) for n in plt.get_fignums()]
    for fig in figs:
        fig.savefig(pp, format='pdf')
    pp.close()


def writeDict(dict, filename, sep=','):
    with open(filename, "w") as f:
        for i in dict.keys():
            f.write('elec[\'' + i + '\'] = ' + str(dict[i]) + '\n')


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

#    # Find start-end peaks
#    # parameter refractory time between peaks
#    refract_time = cycles * 2
#    # parameter number standard deviation above noise
#    points = np.where(norm_sig > (std_parameter * std_gauss_fit))[0]
#    refract_points = np.where(np.diff(points) > refract_time)[0]
#    if not np.size(refract_points):
#        start_peaks = points[0]
#    else:
#        start_peaks = np.array((points[0], points[refract_points + 1]))
#        # end_peaks = np.array((points[refract_points], points[-1]))

    # Find start end of envelope oscillation
    # parameter refractory time between env.oscs.
    refract_time = cycles
    points = np.where(envelope_oscillations > 0)[0]
    refract_points = np.where(np.diff(points) > refract_time)[0]
    start_osc = np.concatenate(([points[0]], points[refract_points+1]))
    end_osc = np.concatenate((points[refract_points], [points[-1]]))

#    start_osc = []
#    end_osc = []
#    for i, peak in enumerate(start_peaks):
#        for j, env in enumerate(start_env):
#            if peak in np.arange(env, end_env[j]):
#                start_osc.append(start_env[i])
#                end_osc.append(end_env[j])

    active_state = np.ones(envelope_oscillations.shape)
    for i, osc in enumerate(start_osc):
        active_state[osc:end_osc[i]] = 0

    return energy, active_state


def calc_active_state_elec(signals, lower_band):

    active_state_all = np.zeros((signals.shape))
    energy_all = np.zeros((signals.shape))

    for i in range(signals.shape[1]):
        signal = signals[:, i]
        energy, active_state = calculate_active_state(signal,
                                                      lower_band)

        active_state_all[:active_state.shape[0], i] = active_state

        energy_all[:energy.shape[0], i] = energy

    return energy_all, active_state_all


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
                           std_parameter=0):
    """
    Calculates active state of a signal taking positive values over a th
    into account.
    Input: Multidimensional signal
    Output: Signal_response
    """
    active_state_fmri = np.zeros((signal.shape))

    std = np.std(signal[:])
    active_state_fmri[np.where(signal > std_parameter*std)] = 1

    return active_state_fmri


def get_lower_band(rithm):

    rithms_dict = dict({'filtered': 0.05,
                        'delta': 0.5,
                        'theta': 3,
                        'alpha': 7,
                        'beta': 13,
                        'gamma': 30,
                        'gamma_high': 70})

    return rithms_dict[rithm]


def figure_3():

    rithms = ['filtered', 'delta', 'theta', 'alpha', 'beta', 'gamma', 'gamma_high']
    SUBJECT_LIST = ['sub-001', 'sub-002', 'sub-003', 'sub-004']
    SESSION_LIST = ['ses-presurg']

    sub_ses_comb = [[subject, session] for subject in SUBJECT_LIST
                    for session in SESSION_LIST]

    sphere = 3
    denoise_type = 'gsr'
    elec_reg_type = 'not_regressed'

    figures_fmri = []

    for i, sub_ses in enumerate(sub_ses_comb):
        sub, ses = sub_ses
        output_dir_path = opj(CWD, 'reports', 'figures', sub)
        if not os.path.exists(output_dir_path):
            os.makedirs(output_dir_path)
        # FUNCTION MATRIX
        # load function (conn matrix?)
        elec_file = opj(DATA, 'raw', 'bids', sub, 'electrodes',
                        'elec.loc')
        elec_location_mni09 = load_elec_file(elec_file)
        ordered_elec = order_dict(elec_location_mni09)
        elec_tags = list(ordered_elec.keys())

        func_file = opj(DATA, 'processed', 'fmriprep', sub, ses, 'func',
                        'time_series_noatlas_' + denoise_type + '_' +
                        str(sphere) + '.txt')
        func_data = np.loadtxt(func_file)

        result = calc_active_state_fmri(func_data)

        plot_active_state(func_data, result, elec_tags)
        ax = plt.title('Subject ' + str(sub) +
                       ' fmri active state ')
        plt.xlabel('time')
        plt.ylabel('tags')
        fig = ax.get_figure()
        figures_fmri.append(fig)
        plt.close()

        dictionary = OrderedDict(zip(elec_tags,
                                     np.sum(result, axis=0, dtype='int32')))

        output_file_fmri = opj(CWD, 'reports', 'stats', 'active_state',
                               'fmri',
                               'stats_fmri_active_state_' + sub)
        writeDict(dictionary, output_file_fmri)

        # Electrophysiology
        input_path = opj(CWD, 'data', 'processed', 'elec_record',
                         sub, 'interictal_' + elec_reg_type)

        random_data = np.load(opj(input_path, 'alpha',
                                  'interictal_1.npy'))
        contact_num = random_data.shape[1]
        all_active_state = np.zeros((12, contact_num))

        for rithm in rithms:
            # load random file
            files = [file for file in os.listdir(opj(input_path, rithm))
                     if file.endswith('npy')]
            for i, file in enumerate(files):
                lower_band = get_lower_band(rithm)
                elec_data = np.load(opj(input_path, rithm, file))
                _, active_state = calc_active_state_elec(elec_data,
                                                         lower_band=lower_band)

                all_active_state[i, :] = np.sum(active_state, axis=0)

            dictionary = OrderedDict(zip(elec_tags,
                                         np.mean(all_active_state,
                                                 axis=0,
                                                 dtype='int32')))
            output_file_elec = opj(CWD, 'reports', 'stats', 'active_state',
                                   'elec',
                                   'stats_elec_active_state_' + sub +
                                   '_' + rithm)
            writeDict(dictionary, output_file_elec)

    multipage(opj('/home/asier/git/ruber/reports/figures/',
                  'fmri_active_state.pdf'),
              figures_fmri,
              dpi=250)
    plt.close("all")


def get_max_active_state_elecs(active_state_dict, amount=10):
    return sorted(active_state_dict, key=active_state_dict.get)[::-1][:amount]


def analyze_results_active_state():

    rithms = ['filtered', 'delta', 'theta', 'alpha', 'beta', 'gamma']
    SUBJECT_LIST = ['sub-001', 'sub-002', 'sub-003', 'sub-004']
    SESSION_LIST = ['ses-presurg']

    sub_ses_comb = [[subject, session] for subject in SUBJECT_LIST
                    for session in SESSION_LIST]

    for i, sub_ses in enumerate(sub_ses_comb):
        sub, ses = sub_ses
        output_dir_path = opj(CWD, 'reports', 'figures', sub)
        if not os.path.exists(output_dir_path):
            os.makedirs(output_dir_path)
        # FUNCTION MATRIX
        # load function (conn matrix?)

        input_file_fmri = opj(CWD, 'reports', 'stats', 'active_state',
                              'fmri',
                              'stats_fmri_active_state_' + sub)

        fmri_active_state = load_elec_file(input_file_fmri)

        fmri_result = get_max_active_state_elecs(fmri_active_state)
        print("Subject {}: \nfMRI active elecs {}.\n".format(sub,
                                                             fmri_result))

        for rithm in rithms:
            input_file_elec = opj(CWD, 'reports', 'stats', 'active_state',
                                  'elec',
                                  'stats_elec_active_state_' + sub +
                                  '_' + rithm)
            elec_active_state = load_elec_file(input_file_elec)
            elec_result = get_max_active_state_elecs(elec_active_state)
            print("{} band electrophysiology active elecs {}.\n".format(rithm,
                  elec_result))


def get_electrode_locations(sub):
    from src.postproc.utils import load_elec_file

    elec_file = opj(CWD, 'data/raw/bids/', sub, 'electrodes/elec.loc')
    elec_dict = load_elec_file(elec_file)
    locations = np.squeeze(np.array(list(elec_dict.values())))

    return locations


def add_electrodes_to_statmap(statmap, sub):
    import nibabel as nib

    statmap_data = statmap.get_data()
    locations = get_electrode_locations(sub)

    for location in locations:
        x, y, z = location
        statmap_data[x, y, z] = 2

    statmap = nib.Nifti1Image(statmap_data,
                              affine=statmap.affine)
    return statmap


def get_max_rois_statmap(rois, num_rois=10):
    import nibabel as nib

    atlas = opj(CWD, 'data/external/bha_atlas_2514_1mm_mni09c.nii.gz')
    most_active_rois = np.argsort(rois)[::-1][:num_rois] + 1  # +1,starts in 1

    atlas_img = nib.load(atlas)
    atlas_data = atlas_img.get_data()

    result = np.zeros((atlas_data.shape))

    for roi in most_active_rois:
        result[np.where(atlas_data == [roi])] = 1

    statmap = nib.Nifti1Image(result,
                              affine=atlas_img.affine)

    return statmap


def figure_4():
    from nilearn import plotting

    SUBJECT_LIST = ['sub-001', 'sub-002', 'sub-003', 'sub-004']
    SESSION_LIST = ['ses-presurg']
    NUM_ROIS = 10

    sub_ses_comb = [[subject, session] for subject in SUBJECT_LIST
                    for session in SESSION_LIST]

    for i, sub_ses in enumerate(sub_ses_comb):
        sub, ses = sub_ses
        output_dir_path = opj(CWD, 'reports', 'figures', sub)
        if not os.path.exists(output_dir_path):
            os.makedirs(output_dir_path)
        # FUNCTION MATRIX
        func_file = opj(DATA, 'processed', 'fmriprep', sub, ses, 'func',
                        'time_series_atlas_2514.txt')
        func_data = np.loadtxt(func_file)

        fmri_active_state = calc_active_state_fmri(func_data)
        fmri_result = np.sum(fmri_active_state, axis=0, dtype='int32')
        output_file_fmri = opj(CWD, 'reports', 'figures', 'active_state',
                               'fmri_active_state_withelec_' + sub)

        statmap = get_max_rois_statmap(fmri_result, num_rois=NUM_ROIS)

        statmap = add_electrodes_to_statmap(statmap, sub)

        plotting.plot_glass_brain(statmap, threshold=0,
                                  display_mode='lyrz',
                                  output_file=output_file_fmri + '.png')

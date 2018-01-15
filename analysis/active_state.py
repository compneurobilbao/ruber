# -*- coding: utf-8 -*-
from src.env import DATA
from src.postproc.utils import load_elec_file, order_dict
from analysis.fig1_fig2_and_stats import plot_matrix, multipage

import os
from os.path import join as opj
import numpy as np
import scipy.io as sio
from matplotlib import pyplot as plt

RAW_ELEC = opj(DATA, 'raw', 'elec_record')
PROCESSED_ELEC = opj(DATA, 'processed', 'elec_record')

SUBJECTS = ['sub-001', 'sub-002', 'sub-003', 'sub-004']
RITHMS = ['prefiltered', 'filtered', 'delta', 'theta',
          'alpha', 'beta', 'gamma', 'gamma_high']

CWD = os.getcwd()


def active_state_to_signal(as_data, real_data):
    """
    Takes real data matrix and active state data in Paolo's format.
    Creates a 0-1 matrix with Paolo's info and filters the real data with it.

    The output is the signal itself, with the non active state points removed
    """

    num_elec = as_data.shape[1]
    output_data = np.zeros((30000, num_elec))

    for elec_idx in range(num_elec):
        for pair in as_data[0][elec_idx]:
            output_data[pair[0]-1:pair[1], elec_idx] = 1

        output_data[:, elec_idx] = (output_data[:, elec_idx] *
                                    real_data[:, elec_idx])

    return output_data


def create_active_state_records():

    for sub in SUBJECTS:
        # create active state folder for sub
        sub_dir = opj(PROCESSED_ELEC, sub, 'active_state')
        if not os.path.exists(sub_dir):
            os.makedirs(sub_dir)

        for rit_num_code, rit in enumerate(RITHMS, 1):

            if rit == 'prefiltered':
                continue
            # create rithm folder in sub
            rithm_dir = opj(sub_dir, rit)
            if not os.path.exists(rithm_dir):
                os.makedirs(rithm_dir)

            for chunk in range(1, 13):  # 12 chunks of 1 sec

                # load real data sub/rithm
                file = opj(PROCESSED_ELEC, sub, 'interictal_not_regressed',
                           rit, 'interictal_' + str(chunk) + '.npy')
                real_data = np.load(file)

                # load AS data
                file = opj(RAW_ELEC, sub, 'interictal',
                           'interictal_' + str(chunk) + '_fb_' +
                           str(rit_num_code) + '_numStd_2.mat')
                as_data = np.array(sio.loadmat(file).get('activeState'))

                as_signal = active_state_to_signal(as_data, real_data)

                output_file = opj(rithm_dir,
                                  'active_state_' + str(chunk) + '.npy')
                np.save(output_file, as_signal)
    return


def create_figures_active_state():
    
    for sub in SUBJECTS:
        output_dir_path = opj(CWD, 'reports', 'figures', 'active_state')
        figures = []
        
        elec_file = opj(DATA, 'raw', 'bids', sub, 'electrodes',
                        'elec.loc')
        elec_location_mni09 = load_elec_file(elec_file)

        ordered_elec = order_dict(elec_location_mni09)

        elec_tags = list(ordered_elec.keys())
        
            
        for rit in RITHMS:
            if rit == 'prefiltered':
                    continue
            files_path = opj(PROCESSED_ELEC, sub, 'active_state', rit)
            
            for num_file, file in enumerate(os.listdir(files_path)):
                if num_file == 0:
                    as_data = np.load(opj(files_path, file))
                    elec_conn_mat = np.corrcoef(as_data.T)
                else:
                    as_data = np.load(opj(files_path, file))
                    elec_conn_mat += np.corrcoef(as_data.T)
            
            elec_conn_mat = elec_conn_mat / (num_file+1)
            
            plot_matrix(elec_conn_mat, elec_tags)
            plt.colorbar()
            ax = plt.title('Active state :' +
                            ' sub: ' + sub +
                            ' rithm: ' + rit)
            fig = ax.get_figure()
            figures.append(fig)
            plt.close()
            
        multipage(opj(output_dir_path,
                      'Active state :' +
                        ' sub: ' + sub +
                        '.pdf'),
                      figures,
                      dpi=250)
        
            
def create_figures_not_active_state():
    
    for sub in SUBJECTS:
        output_dir_path = opj(CWD, 'reports', 'figures', 'active_state')
        figures = []
        
        elec_file = opj(DATA, 'raw', 'bids', sub, 'electrodes',
                        'elec.loc')
        elec_location_mni09 = load_elec_file(elec_file)

        ordered_elec = order_dict(elec_location_mni09)

        elec_tags = list(ordered_elec.keys())
        
            
        for rit in RITHMS:
            if rit == 'prefiltered':
                    continue
            files_path = opj(PROCESSED_ELEC, sub, 'interictal_not_regressed', rit)
            
            for num_file, file in enumerate(os.listdir(files_path)):
                if num_file == 0:
                    if file.endswith('npy'):
                        as_data = np.load(opj(files_path, file))
                        elec_conn_mat = np.corrcoef(as_data.T)
                else:
                    if file.endswith('npy'):
                        as_data = np.load(opj(files_path, file))
                        elec_conn_mat += np.corrcoef(as_data.T)
            
            elec_conn_mat = elec_conn_mat / 12
            
            plot_matrix(elec_conn_mat, elec_tags)
            plt.colorbar()
            ax = plt.title('normal sig :' +
                            ' sub: ' + sub +
                            ' rithm: ' + rit)
            fig = ax.get_figure()
            figures.append(fig)
            plt.close()
        
        multipage(opj(output_dir_path,
                      'normal sig :' +
                        ' sub: ' + sub +
                        '.pdf'),
                      figures,
                      dpi=250)


def calc_distance(point1, point2):
    """
    calc euclidean distance
    """
    point1 = np.array(point1)
    point2 = np.array(point2)
    
    return np.linalg.norm(point1 - point2)
    

def create_distance_matrices():
    
    for sub in SUBJECTS:
        elec_file = opj(DATA, 'raw', 'bids', sub, 'electrodes',
                        'elec.loc')
        elec_location_mni09 = load_elec_file(elec_file)

        ordered_elec = order_dict(elec_location_mni09)
        
        #get num elec and dist_mat
        num_elec = len(ordered_elec)
        dist_mat = np.zeros((num_elec, num_elec))
        for idx_i, elec_pos1 in enumerate(ordered_elec.itervalues()):
            for idx_j, elec_pos2 in enumerate(ordered_elec.itervalues()):
                dist_mat[idx_i, idx_j] = calc_distance(elec_pos1, elec_pos2)

        np.save('file_path', dist_mat)

    return          
            
            
            
            
            
            
            
            
            
            
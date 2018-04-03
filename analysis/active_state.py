# -*- coding: utf-8 -*-
from src.env import DATA
from src.postproc.utils import load_elec_file, order_dict
from analysis.fig1_fig2_and_stats import plot_matrix, multipage
from analysis.bha import cross_modularity

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
        output_dir_path = opj(CWD, 'reports', 'matrices', sub)
        if not os.path.exists(output_dir_path):
            os.makedirs(output_dir_path)
                              
        elec_file = opj(DATA, 'raw', 'bids', sub, 'electrodes',
                        'elec.loc')
        elec_location_mni09 = load_elec_file(elec_file)

        ordered_elec = order_dict(elec_location_mni09)
        
        #get num elec and dist_mat
        num_elec = len(ordered_elec)
        dist_mat = np.zeros((num_elec, num_elec))
        for idx_i, elec_pos1 in enumerate(ordered_elec.values()):
            for idx_j, elec_pos2 in enumerate(ordered_elec.values()):
                dist_mat[idx_i, idx_j] = calc_distance(elec_pos1, elec_pos2)
        
        # Normalize
        dist_mat = dist_mat / np.max(dist_mat)

        np.save(opj(output_dir_path, 'DC.npy'),
                dist_mat)
    return


def corr_mat(X):
    from scipy import stats

    N=X.shape[1]
    rho=np.empty((N,N), dtype=float)
    pval=np.empty((N,N), dtype=float)
    for i in range(N):
        v1=X[:,i]
        for j in range(i,N):
             v2=X[:,j]
             C,P=stats.pearsonr(v1,v2)
             rho[i,j]=C
             rho[j,i]=C
             pval[i,j]=P
             pval[j,i]=P
    return rho, pval


def create_elec_matrices():
    th = 0.01 # threshold for getting just significant pvalues
    for sub in SUBJECTS:
        output_dir_path = opj(CWD, 'reports', 'matrices', sub)
        if not os.path.exists(output_dir_path):
            os.makedirs(output_dir_path)
                          
        for rit in RITHMS:
            if rit == 'prefiltered':
                    continue      

            # Active State
            # files_path = opj(PROCESSED_ELEC, sub, 'active_state', rit)
            
            # Normal
            files_path = opj(PROCESSED_ELEC, sub, 'interictal_not_regressed', rit)

            
            for num_file, file in enumerate(os.listdir(files_path)):
                if num_file == 0:
                    as_data = np.load(opj(files_path, file))
                    total_as_data = as_data
                else:
                    as_data = np.load(opj(files_path, file))
                    total_as_data = np.concatenate((total_as_data, as_data))
                
            elec_conn_mat, pvals = corr_mat(total_as_data)    
            elec_conn_mat[np.where(pvals > th)] = 0
    
            np.save(opj(output_dir_path, 'EL_'+rit+'.npy'),
                    elec_conn_mat)
    return            
            

def create_FC_SC_matrices():
    from nilearn.connectome import ConnectivityMeasure

    MINIMUM_FIBERS = 10

    sphere = 3
    denoise_type = 'gsr'
    ses = 'ses-presurg'
    
    for sub in SUBJECTS:    
        output_dir_path = opj(CWD, 'reports', 'matrices', sub)
        if not os.path.exists(output_dir_path):
            os.makedirs(output_dir_path)
            
        # load function (conn matrix?)
        func_file = opj(DATA, 'processed', 'fmriprep', sub, ses, 'func',
                        'time_series_noatlas_' + denoise_type + '_' +
                        str(sphere) + '.txt')
        func_mat = np.loadtxt(func_file)

        correlation_measure = ConnectivityMeasure(kind='correlation')
        fc_mat = correlation_measure.fit_transform([func_mat])[0]
        
        fc_mat_neg = fc_mat.copy()
        fc_mat_pos = fc_mat.copy()
        
        fc_mat_neg[np.where(fc_mat>0)] = 0
        fc_mat_pos[np.where(fc_mat<0)] = 0

        # STRUCT MATRIX
        sc_mat = np.load(opj(DATA, 'raw', 'bids', sub, 'electrodes', ses,
                             'con_mat_noatlas_' +
                             str(sphere) + '.npy'))
        
        sc_mat_bin = sc_mat.copy()
        sc_mat_bin[np.where(sc_mat_bin > MINIMUM_FIBERS)] = 1
        
        sc_mat = sc_mat / np.max(sc_mat)

        np.save(opj(output_dir_path, 'FC.npy'),
                fc_mat) 
        np.save(opj(output_dir_path, 'FC_NEG.npy'),
                fc_mat_neg) 
        np.save(opj(output_dir_path, 'FC_POS.npy'),
                fc_mat_pos) 
        np.save(opj(output_dir_path, 'SC.npy'),
                sc_mat)             
        np.save(opj(output_dir_path, 'SC_BIN.npy'),
                        sc_mat_bin)    
         
def modularity_analysis():

    from scipy import spatial, cluster
    from itertools import product         
    
    S_T = [['SC', 'FC_POS'],
           ['DC', 'FC_POS'],['EL_alpha', 'FC_POS'],['EL_beta', 'FC_POS'],
           ['FC_POS', 'EL_alpha'],['FC_POS', 'EL_beta']]
    ALPHA = 0.45
    BETA = 0.0
    MAX_CLUSTERS = 50
    output_dir = opj(CWD, 'reports', 'figures', 'active_state')
    
    figures = []

    for sub in SUBJECTS:
        input_dir_path = opj(CWD, 'reports', 'matrices', sub)
        legend = []
        for source, target in S_T:
            
            source_network =  np.load(opj(input_dir_path, source + '.npy'))
            target_network = np.load(opj(input_dir_path, target + '.npy'))
            legend.append(source + ' -> ' + target)

            result = np.zeros(MAX_CLUSTERS)
            
            for num_clusters in range(2,MAX_CLUSTERS):
                """
                Source dendogram -> target follows source
                """
                
                    
                if source in ['SC_BIN']: 
                    # SC_BIN discarded for the moment. 
                    # TODO: Calculation of Y
                    Z = cluster.hierarchy.linkage(Y, method='average')
                    T = cluster.hierarchy.cut_tree(Z,  n_clusters=num_clusters)
                    Xsf, Qff, Qsf, Lsf = cross_modularity(target_network,
                                                          source_network,
                                                          ALPHA,
                                                          BETA,
                                                          T[:, 0])
                    result[num_clusters] = np.nan_to_num(Xsf)
                else:
                    Y = spatial.distance.pdist(source_network, metric='cosine')
                    Y = np.nan_to_num(Y)
                    Z = cluster.hierarchy.linkage(Y, method='weighted')
                    T = cluster.hierarchy.cut_tree(Z, n_clusters=num_clusters)
                
                    Xsf, Qff, Qsf, Lsf = cross_modularity(target_network,
                                                          source_network,
                                                          ALPHA,
                                                          BETA,
                                                          T[:, 0])
                    result[num_clusters] = np.nan_to_num(Xsf)

                            
            plt.plot(result)
            plt.hold(True)
        plt.legend(legend)
        plt.legend(bbox_to_anchor=(1.04,1), loc="upper left")
        plt.xlabel('# clusters')
        plt.ylabel('modularity value')
        plt.ylim((0, 0.5))
        ax = plt.title('Modularity_' + sub )
        fig = ax.get_figure()
        figures.append(fig)
        plt.close()
    
    multipage(opj(output_dir,
                  'xmod_2:fcpos.pdf'),
                    figures,
                    dpi=250)
    
        
def single_link_analysis():

    from itertools import combinations       
    
    MODALITIES = ['SC', 'DC', 'SC_BIN', 'FC', 'FC_POS', 'FC_NEG',
                  'EL_filtered', 'EL_delta', 'EL_theta', 'EL_alpha', 'EL_beta',
                  'EL_gamma', 'EL_gamma_high', 
                  'EL_AS_filtered', 'EL_AS_delta', 'EL_AS_theta', 'EL_AS_alpha',
                  'EL_AS_beta', 'EL_AS_gamma', 'EL_AS_gamma_high']
    MOD_IDX = {v: k for k, v in dict(enumerate(MODALITIES)).items()}
    num_mod = len(MOD_IDX)
    output_dir = opj(CWD, 'reports', 'figures', 'active_state')
    
    result_mat = np.zeros((num_mod, num_mod))

    figures = []

    for sub in SUBJECTS:
        input_dir_path = opj(CWD, 'reports', 'matrices', sub)

        for source, target in combinations(MODALITIES, 2):
            
            arr_1 =  np.load(opj(input_dir_path, source + '.npy')).flatten()
            arr_2 = np.load(opj(input_dir_path, target + '.npy')).flatten()

            if arr_1 in ['SC', 'SC_BIN']:
                idx = np.where(arr_1==0)
                arr_1 = np.delete(arr_1, idx)
                arr_2 = np.delete(arr_2, idx)
            
            result_mat[MOD_IDX[source],MOD_IDX[target]] = np.corrcoef(arr_1, arr_2)[0][1]       
        
        plot_matrix(result_mat.T, MODALITIES)
        plt.clim(-1,1)
        plt.colorbar(orientation="horizontal")
        ax = plt.title('Single_link_' + sub )
        fig = ax.get_figure()
        figures.append(fig)
        plt.close()

    multipage(opj(output_dir,
                  'Single_link.pdf'),
                    figures,
                    dpi=250)                 
                    

def single_link_analysis_mean_subjects():

    from itertools import combinations       
    
    MODALITIES = ['SC', 'DC', 'SC_BIN', 'FC', 'FC_POS', 'FC_NEG',
                  'EL_filtered', 'EL_delta', 'EL_theta', 'EL_alpha', 'EL_beta',
                  'EL_gamma', 'EL_gamma_high', 
                  'EL_AS_filtered', 'EL_AS_delta', 'EL_AS_theta', 'EL_AS_alpha',
                  'EL_AS_beta', 'EL_AS_gamma', 'EL_AS_gamma_high']
    MOD_IDX = {v: k for k, v in dict(enumerate(MODALITIES)).items()}
    num_mod = len(MOD_IDX)
    output_dir = opj(CWD, 'reports', 'figures', 'active_state')
    
    result_mat = np.zeros((num_mod, num_mod))
    result_mat_all = np.zeros((num_mod, num_mod, 4))

    figures = []

    for i,sub in enumerate(SUBJECTS):
        input_dir_path = opj(CWD, 'reports', 'matrices', sub)

        for source, target in combinations(MODALITIES, 2):
            
            arr_1 =  np.load(opj(input_dir_path, source + '.npy')).flatten()
            arr_2 = np.load(opj(input_dir_path, target + '.npy')).flatten()

            if arr_1 in ['SC', 'SC_BIN']:
                idx = np.where(arr_1==0)
                arr_1 = np.delete(arr_1, idx)
                arr_2 = np.delete(arr_2, idx)
            
            result_mat[MOD_IDX[source],MOD_IDX[target]] = np.corrcoef(arr_1, arr_2)[0][1]       
        
        result_mat_all[:,:,i] = result_mat
        plot_matrix(result_mat.T, MODALITIES)
        plt.clim(-1,1)
        plt.colorbar(orientation="horizontal")
        ax = plt.title('Single_link_' + sub )
        fig = ax.get_figure()
        figures.append(fig)
        plt.close()

    multipage(opj(output_dir,
                  'Single_link.pdf'),
                    figures,
                    dpi=250)       
                    
                    
plot_matrix(np.mean(result_mat_all, 2).T, MODALITIES)
plt.clim(-1,1)
ax = plt.title('Mean similiraty matrix')
plt.savefig('/home/asier/Desktop/fig2.eps', format='eps', dpi=300)

fig = ax.get_figure()
figures.append(fig)
plt.close()                    
                    
   


def plot_matrix2(matrix, elec_tags, log=False):
    plt.figure(figsize=(10, 10))
    # Mask the main diagonal for visualization:

    if log:
        matrix = log_transform(matrix)

    plt.imshow(matrix, interpolation="nearest", cmap="RdBu_r")
    # vmax=0.8, vmin=-0.8)

    # Add labels and adjust margins
    plt.xticks(range(len(elec_tags)), elec_tags, rotation=90, fontweight='bold')
    plt.yticks(range(len(elec_tags)), elec_tags, fontweight='bold')
    plt.gca().yaxis.tick_right()
    plt.subplots_adjust(left=.01, bottom=.3, top=.99, right=.62)

                 
N = 6
b =np.random.randint(0,1000,(N,N))
mat = (b + b.T)/2         
mat[np.diag_indices_from(mat)] = 1
MODALITIES = ["CA1", "BLA", "mPFC", "S1", "M1", "PRh" ]
plot_matrix2(mat, MODALITIES)
plt.clim(0,1000)
plt.colorbar(orientation="horizontal")
plt.ylabel('regions', fontweight='bold')
plt.xlabel('regions', fontweight='bold')
plt.savefig('/home/asier/Desktop/SC.eps', format='eps', dpi=300)



ax = plt.title('FC' )
plt.close()             

N = 6
b = np.random.random_integers(-1,1,size=(N,N))
mat = (b + b.T)/2         
mat[np.diag_indices_from(mat)] = 1
MODALITIES = ["CA1", "BLA", "mPFC", "S1", "M1", "PRh" ]
plot_matrix(mat, MODALITIES)
plt.clim(-1,1)
plt.colorbar(orientation="horizontal")
ax = plt.title('FC' )
plt.close()                         
  
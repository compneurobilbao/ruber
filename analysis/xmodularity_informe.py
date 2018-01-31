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


def modularity_analysis():

    from scipy import spatial, cluster
    from itertools import product         
    
    SOURCES = ['FC']
    TARGETS = ['SC', 'EL_theta', 'EL_alpha', 'EL_beta']
    ALPHA = 0.45
    BETA = 0.0
    MAX_CLUSTERS = 20
    output_dir = opj(CWD, 'reports', 'figures', 'active_state')
    
    figures = []

    for sub in SUBJECTS:
        input_dir_path = opj(CWD, 'reports', 'matrices', sub)
        legend = []
        for source, target in product(SOURCES, TARGETS):
            
            source_network =  np.load(opj(input_dir_path, source + '.npy'))
            target_network = np.load(opj(input_dir_path, target + '.npy'))
            legend.append(source + ' -> ' + target)

            result = np.zeros(MAX_CLUSTERS)
            
            for num_clusters in range(2,MAX_CLUSTERS):
                """
                Source dendogram -> target follows source
                """
                Y = spatial.distance.pdist(source_network, metric='cosine')
                Y = np.nan_to_num(Y)
                Z = cluster.hierarchy.linkage(Y, method='weighted')
                T = cluster.hierarchy.cut_tree(Z, n_clusters=num_clusters)[:, 0]
            


                result[num_clusters] = np.nan_to_num(Xsf)

                            
#            plt.plot(result)
#            plt.hold(True)
#        plt.legend(legend)
#        plt.legend(bbox_to_anchor=(1.04,1), loc="upper left")
#        plt.xlabel('# clusters')
#        plt.ylabel('modularity value')
#        plt.ylim((0, 0.5))
#        ax = plt.title('Modularity_' + sub )
#        fig = ax.get_figure()
#        figures.append(fig)
#        plt.close()
#    
#    multipage(opj(output_dir,
#                  'xmod_el_2_fmri_pos.pdf'),
#                    figures,
#                    dpi=250)
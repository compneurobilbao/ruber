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

CWD = os.getcwd()

# TODO: target tags from informe
target_tags = [ 'OIL4', 'OIL5', 'OIL6', 'OIL7',
       'OIL8', 'OIL9',  'TI1', 'TI5',]

def modularity_analysis():

    from scipy import spatial, cluster
    from itertools import product         
    
    SOURCES = ['FC', 'SC', 'EL_theta', 'EL_alpha', 'EL_beta']

    MAX_CLUSTERS = 20
#    output_dir = opj(CWD, 'reports', 'figures', 'active_state')
#    
#    figures = []
    result = np.zeros((len(SUBJECTS),MAX_CLUSTERS, MAX_CLUSTERS))


    for idx_sub, sub in enumerate(SUBJECTS):
        input_dir_path = opj(CWD, 'reports', 'matrices', sub)
        elec_file = opj(DATA, 'raw', 'bids', sub, 'electrodes',
                        'elec.loc')
        elec_location_mni09 = load_elec_file(elec_file)
        ordered_elec = order_dict(elec_location_mni09)
        elec_tags = np.array(list(ordered_elec.keys()))

        for source in SOURCES:
            
            source_network =  np.load(opj(input_dir_path, source + '.npy'))

            
            for num_clusters in range(2,MAX_CLUSTERS):
                """
                Source dendogram -> target follows source
                """
                Y = spatial.distance.pdist(source_network, metric='cosine')
                Y = np.nan_to_num(Y)
                Z = cluster.hierarchy.linkage(Y, method='weighted')
                T = cluster.hierarchy.cut_tree(Z, n_clusters=num_clusters)[:, 0]
                
                for clust in range(num_clusters):
                    idx = np.where(T==clust)
                    clust_tags = elec_tags[idx[0]]
                    matching = set(clust_tags) & set(target_tags)
                    result[idx_sub, num_clusters, clust] = len(matching) / len(target_tags) * 100
                    
                            
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
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

# Obtained from informe
target_tags_dict = {'sub-001': ['OIL1', 'OIL2', 'OIL3', 'OIL4', 'OIL5', 'OIL6',
                                'TI1', 'TI2', 'OIM1', 'OIM2', 'OIM3',],
                    'sub-002': ['A2', 'A3', 'A4', 'C2', 'C3', 'C4', 'C5'],
                    'sub-003': ['B1', 'B2', 'C1', 'C2', 'C3', 'C4'],
                    'sub-004': ['D6', 'D7', 'D8', 'D9', 'D10', 'C5', 'C6', 'C7', 'C8'],
                    }


def calc_clust_similarity(clust_size, target_size):
    """ 
    Linear function to calculate how close the size of the cluster
    is to the real epileptogenic cluster that we want to find
    """
    
    if clust_size > target_size:
        if clust_size >= 2*target_size:
            clust_size = 0
        else:
            clust_size = target_size - (clust_size - target_size)
        
    return (clust_size/target_size)


def modularity_analysis():

    from scipy import spatial, cluster
    
    SOURCES = ['FC', 'SC', 'EL_theta', 'EL_alpha', 'EL_beta']

    MAX_CLUSTERS = 40
    output_dir = opj(CWD, 'reports', 'figures', 'active_state')
    
    figures = []
    
    for sub in SUBJECTS:
        input_dir_path = opj(CWD, 'reports', 'matrices', sub)
        elec_file = opj(DATA, 'raw', 'bids', sub, 'electrodes',
                        'elec.loc')
        elec_location_mni09 = load_elec_file(elec_file)
        ordered_elec = order_dict(elec_location_mni09)
        elec_tags = np.array(list(ordered_elec.keys()))
        
        target_tags = target_tags_dict[sub]

        for source in SOURCES:
            
            source_network =  np.load(opj(input_dir_path, source + '.npy'))

            result = np.zeros((MAX_CLUSTERS, MAX_CLUSTERS))
            for num_clusters in range(1,MAX_CLUSTERS):
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
                    
                    clust_sim = calc_clust_similarity(len(target_tags),
                                                      len(clust_tags))
                    result[num_clusters, clust] = (len(matching) / len(target_tags)) * clust_sim
                    
                            
            plot_matrix(result, range(MAX_CLUSTERS))
            plt.clim(0,1)
            plt.colorbar(orientation="vertical")
            plt.ylabel('# clusters - partition')
            plt.xlabel('# of specific cluster')
            ax = plt.title('Informe match ' + sub + ' ' + source )
            fig = ax.get_figure()
            figures.append(fig)
            plt.close()

    multipage(opj(output_dir,
                  'Informe match.pdf'),
                    figures,
                    dpi=250)

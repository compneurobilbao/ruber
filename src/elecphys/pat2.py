# -*- coding: utf-8 -*-
import os
from os.path import join as opj
import numpy as np


from src.elecphys.utils import (regress_signal,
                                filter_and_save_all_bands,
                                )

"""
This code is to load the elec.loc file and order the tags as in the 
conn. matrixes.
The issue here is to order the electrodes in the same order as them. 
Compare the ordered_elec
/home/asier/git/ruber/data/raw/bids/sub-002/electrodes/elec.loc
"""
#from src.postproc.utils import load_elec_file, order_dict
#from src.env import DATA
#sub = 'sub-002'
#elec_file = opj(DATA, 'raw', 'bids', sub, 'electrodes',
#                            'elec.loc')
#elec_location_mni09 = load_elec_file(elec_file)
#ordered_elec = order_dict(elec_location_mni09)
#elec_tags = list(ordered_elec.keys())


def reorder_pat2_elec(elec_data):

    elec_data_ordered = np.zeros(elec_data.shape, dtype='float16')

    # A
    elec_data_ordered[:, :5] = elec_data[:, :5]
    # B
    elec_data_ordered[:, 15:20] = elec_data[:, 5:10]
    # C
    elec_data_ordered[:, 20:28] = elec_data[:, 10:18]
    # D
    elec_data_ordered[:, 28:33] = elec_data[:, 18:23]
    # E
    elec_data_ordered[:, 33:41] = elec_data[:, 23:31]
    # F
    elec_data_ordered[:, 41:46] = elec_data[:, 31:36]
    # G
    elec_data_ordered[:, 46:56] = elec_data[:, 36:46]
    # AM
    elec_data_ordered[:, 5:15] = elec_data[:, 46:56]
    # HP
    elec_data_ordered[:, 56:] = elec_data[:, 56:]

    return elec_data_ordered


def reorder_and_regress_pat2_elec(elec_data):

    elec_data_ordered = np.zeros(elec_data.shape, dtype='float16')

    # A
    elec_data_ordered[:, :5] = regress_signal(elec_data[:, :5])
    # B
    elec_data_ordered[:, 15:20] = regress_signal(elec_data[:, 5:10])
    # C
    elec_data_ordered[:, 20:28] = regress_signal(elec_data[:, 10:18])
    # D
    elec_data_ordered[:, 28:33] = regress_signal(elec_data[:, 18:23])
    # E
    elec_data_ordered[:, 33:41] = regress_signal(elec_data[:, 23:31])
    # F
    elec_data_ordered[:, 41:46] = regress_signal(elec_data[:, 31:36])
    # G
    elec_data_ordered[:, 46:56] = regress_signal(elec_data[:, 36:46])
    # AM
    elec_data_ordered[:, 5:15] = regress_signal(elec_data[:, 46:56])
    # HP
    elec_data_ordered[:, 56:] = regress_signal(elec_data[:, 56:])

    return elec_data_ordered


def create_pat2_files():

    input_path = '/home/asier/git/ruber/data/raw/elec_record/sub-002/interictal'
    out_path_reg = '/home/asier/git/ruber/data/processed/elec_record/sub-002/interictal_regressed'
    out_path_not_reg = '/home/asier/git/ruber/data/processed/elec_record/sub-002/interictal_not_regressed'

    for file in os.listdir(input_path):
        print(file)
        elec_data = np.load(opj(input_path, file))
        elec_data_ordered = reorder_pat2_elec(elec_data)
        elec_data_regressed = reorder_and_regress_pat2_elec(elec_data)

        filter_and_save_all_bands(out_path_not_reg, elec_data_ordered, file)
        filter_and_save_all_bands(out_path_reg, elec_data_regressed, file)

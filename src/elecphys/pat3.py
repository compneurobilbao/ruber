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
#sub = 'sub-003'
#elec_file = opj(DATA, 'raw', 'bids', sub, 'electrodes',
#                            'elec.loc')
#elec_location_mni09 = load_elec_file(elec_file)
#ordered_elec = order_dict(elec_location_mni09)
#elec_tags = list(ordered_elec.keys())


def reorder_and_regress_pat3_elec(elec_data):

    elec_data_ordered = np.zeros(elec_data.shape, dtype='float16')

    # A
    elec_data_ordered[:, :12] = regress_signal(elec_data[:, :12])
    # B
    elec_data_ordered[:, 12:20] = regress_signal(elec_data[:, 12:20])
    # C
    elec_data_ordered[:, 20:28] = regress_signal(elec_data[:, 20:28])

    return elec_data_ordered


def create_pat3_files():

    input_path = '/home/asier/git/ruber/data/raw/elec_record/sub-003/interictal'
    out_path_reg = '/home/asier/git/ruber/data/processed/elec_record/sub-003/interictal_regressed'
    out_path_not_reg = '/home/asier/git/ruber/data/processed/elec_record/sub-003/interictal_not_regressed'

    for file in os.listdir(input_path):
        print(file)
        elec_data = np.load(opj(input_path, file))
        elec_data_ordered = elec_data  # no need of reorder
        elec_data_regressed = reorder_and_regress_pat3_elec(elec_data)

        filter_and_save_all_bands(out_path_not_reg, elec_data_ordered, file)
        filter_and_save_all_bands(out_path_reg, elec_data_regressed, file)

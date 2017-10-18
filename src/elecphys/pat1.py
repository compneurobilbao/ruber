# -*- coding: utf-8 -*-
import os
from os.path import join as opj
import numpy as np


from src.elecphys.utils import (regress_signal,
                                filter_and_save_all_bands,
                                )


def reorder_pat1_elec(elec_data):

    elec_data_ordered = np.zeros(elec_data.shape, dtype='float16')

    # OIM
    elec_data_ordered[:, 24:36] = elec_data[:, :12]
    # OIL
    elec_data_ordered[:, 12:24] = elec_data[:, 12:24]
    # OSM
    elec_data_ordered[:, 41:49] = elec_data[:, 24:32]
    # OSL
    elec_data_ordered[:, 36:41] = elec_data[:, 32:37]
    # TI
    elec_data_ordered[:, 49:57] = elec_data[:, 37:45]
    # A
    elec_data_ordered[:, :12] = elec_data[:, 45:57]

    return elec_data_ordered


def reorder_and_regress_pat1_elec(elec_data):

    elec_data_ordered = np.zeros(elec_data.shape, dtype='float16')

    # OIM
    elec_data_ordered[:, 24:36] = regress_signal(elec_data[:, :12])
    # OIL
    elec_data_ordered[:, 12:24] = regress_signal(elec_data[:, 12:24])
    # OSM
    elec_data_ordered[:, 41:49] = regress_signal(elec_data[:, 24:32])
    # OSL
    elec_data_ordered[:, 36:41] = regress_signal(elec_data[:, 32:37])
    # TI
    elec_data_ordered[:, 49:57] = regress_signal(elec_data[:, 37:45])
    # A
    elec_data_ordered[:, :12] = regress_signal(elec_data[:, 45:57])

    return elec_data_ordered


def create_pat1_files():

    input_path = '/home/asier/git/ruber/data/raw/elec_record/sub-001/interictal'
    out_path_reg = '/home/asier/git/ruber/data/processed/elec_record/sub-001/interictal_regressed'
    out_path_not_reg = '/home/asier/git/ruber/data/processed/elec_record/sub-001/interictal_not_regressed'

    for file in os.listdir(input_path):
        print(file)
        elec_data = np.load(opj(input_path, file))
        elec_data_ordered = reorder_pat1_elec(elec_data)
        elec_data_regressed = reorder_and_regress_pat1_elec(elec_data)

        filter_and_save_all_bands(out_path_not_reg, elec_data_ordered, file)
        filter_and_save_all_bands(out_path_reg, elec_data_regressed, file)

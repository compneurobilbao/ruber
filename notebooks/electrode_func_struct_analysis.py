# -*- coding: utf-8 -*-

from src.postproc.utils import load_elec_file
from src.env import DATA, ATLAS_TYPES
#import os.path as op
from os.path import join as opj
import numpy as np


SUBJECT_LIST = ['sub-001']
SESSION_LIST = ['ses-presurg']


def order_dict(dictionary):
    import collections
    import re

    my_fun = lambda k, v: [k, int(v)]

    ordered = collections.OrderedDict(
            sorted(dictionary.items(),
                   key=lambda t: my_fun(*re.match(r'([a-zA-Z]+)(\d+)',
                                                  t[0]).groups())))
    return ordered


if __name__ == "__main__":

    sub_ses_comb = [[subject, session] for subject in SUBJECT_LIST
                    for session in SESSION_LIST]

    for sub, ses in sub_ses_comb:
        for atlas in ATLAS_TYPES:
            # load ROI location of each contact
            elec_file = opj(DATA, 'raw', 'bids', sub, 'electrodes',
                            'sub-001_elec_2514_closest_rois.roi')
            elec_location_mni09 = load_elec_file(elec_file)

            ordered_elec = order_dict(elec_location_mni09)
    
            elec_tags = list(ordered_elec.keys())
            elec_rois = np.array(list(ordered_elec.values()))[:,0]

            idx = np.ix_(elec_rois, elec_rois)
            
            # load function (conn matrix?)
            
            
            # load struct
            struct_file = opj(DATA, 'processed', 'tract', '_session_id_' +
                              ses + '_subject_id_' + sub,
                              'conmat_' + atlas + '_sc.csv')
            
            struct_mat = np.loadtxt(struct_file, delimiter = ',', skiprows=1)

            struct_mat[idx]



from matplotlib import pyplot as plt

plt.figure(figsize=(10, 10))
# Mask the main diagonal for visualization:

plt.imshow(struct_mat[idx], interpolation="nearest", cmap="RdBu_r")
           #vmax=0.8, vmin=-0.8)

# Add labels and adjust margins
x_ticks = plt.xticks(range(len(elec_tags)), elec_tags, rotation=90)
y_ticks = plt.yticks(range(len(elec_tags)), elec_tags)
plt.gca().yaxis.tick_right()
plt.subplots_adjust(left=.01, bottom=.3, top=.99, right=.62)



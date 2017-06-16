# -*- coding: utf-8 -*-

DATA = '/home/asier/git/ruber/data/'
RAW_DATA = '/home/asier/git/ruber/data/raw'
BIDS_DATA = '/home/asier/git/ruber/data/raw/bids'
HEUDICONV_BIN = '/home/asier/git/ruber/src/heudiconv/heudiconv'
HEUDICONV_FOLDER = '/home/asier/git/ruber/src/heudiconv'
SESSION_TYPES = ['presurg', 'postsurg1', 'postsurg2']
ATLAS_TYPES = ['atlas_2514', 'atlas_2754']

CONFOUNDS_ID = ['FramewiseDisplacement',
                'aCompCor0',
                'aCompCor1',
                'aCompCor2',
                'aCompCor3',
                'aCompCor4',
                'aCompCor5',
                'X',
                'Y',
                'Z',
                'RotX',
                'RotY',
                'RotZ',
                ]

NEIGHBOURS = [0, 1]
# TODO: this as iterable variable
ELECTRODE_KERNEL_SIZE = [1, 2, 3]  # Gaussian kernel (sigma in mm, not voxels)
FRAMEWISE_DISP_THRES = 0.2  # For scrubbing

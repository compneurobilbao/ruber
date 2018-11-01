# -*- coding: utf-8 -*-

DATA = '/home/asier/git/ruber/data'
RAW_DATA = '/home/asier/git/ruber/data/raw'
BIDS_DATA = '/home/asier/git/ruber/data/raw/bids'
HEUDICONV_BIN = '/home/asier/git/ruber/src/heudiconv/heudiconv'
HEUDICONV_FOLDER = '/home/asier/git/ruber/src/heudiconv'


#DATA = '/ems/elsc-labs/mezer-a/asier.erramuzpe/Desktop/ruber/data/'
#RAW_DATA = '/ems/elsc-labs/mezer-a/asier.erramuzpe/Desktop/ruber/data/raw'
#BIDS_DATA = '/ems/elsc-labs/mezer-a/asier.erramuzpe/Desktop/data/raw/bids'
#HEUDICONV_BIN = '/ems/elsc-labs/mezer-a/asier.erramuzpe/Desktop/src/heudiconv/heudiconv'
#HEUDICONV_FOLDER = '/ems/elsc-labs/mezer-a/asier.erramuzpe/Desktop/src/heudiconv'


SESSION_TYPES = ['presurg', 'postsurg1', 'postsurg2']
ATLAS_TYPES = ['atlas_2514', 'atlas_2754']

CONFOUNDS_ID = ['FramewiseDisplacement',
                'WhiteMatter',
                'GlobalSignal',
                'X',
                'Y',
                'Z',
                'RotX',
                'RotY',
                'RotZ',
                ]

#CONFOUNDS_ID = ['FramewiseDisplacement',
#                'aCompCor0',
#                'aCompCor1',
#                'aCompCor2',
#                'aCompCor3',
#                'aCompCor4',
#                'aCompCor5',
#                'X',
#                'Y',
#                'Z',
#                'RotX',
#                'RotY',
#                'RotZ',
#                ]

NEIGHBOURS = [0, 1]
# TODO: this as iterable variable and convert pipeline to nipype
ELECTRODE_SPHERE_SIZE = [2, 3]
FRAMEWISE_DISP_THRES = 0.2  # For scrubbing

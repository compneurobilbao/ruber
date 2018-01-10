# -*- coding: utf-8 -*-
from src.env import DATA

import os.path as op
from os.path import join as opj
import numpy as np

PROCESSED = opj(DATA, 'processed', 'fmriprep')
EXTERNAL = opj(DATA, 'external')
EXTERNAL_MNI_09c = opj(EXTERNAL, 'standard_mni_asym_09c')


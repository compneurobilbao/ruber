#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 22 15:21:11 2017

@author: asier
"""

import os


def create_key(template, outtype=('nii.gz',), annotation_classes=None):
    if template is None or not template:
        raise ValueError('Template must be a valid format string')
    return template, outtype, annotation_classes


def infotodict(seqinfo):
    """Heuristic evaluator for determining which runs belong where

    allowed template fields - follow python string module:

    item: index within category
    subject: participant id
    seqitem: run number during scanning
    subindex: sub index within group
    """

    t1w = create_key('anat/sub-{subject}_{session}_T1w')
    # fmap = create_key('fmap/sub-{subject}_fieldmap')

    info = {t1w: []}# , fmap: []}

    for s in seqinfo:
        """
        The namedtuple `s` contains the following fields:

        * total_files_till_now
        * example_dcm_file
        * series_number
        * dcm_dir_name
        * unspecified2
        * unspecified3
        * dim1
        * dim2
        * dim3
        * dim4
        * TR
        * TE
        * protocol_name
        * is_motion_corrected
        * is_derived
        * patient_id
        * study_description
        * referring_physician_name
        * series_description
        * image_type
        """
        
        if 'SAG_3D_T1' in s.protocol_name:
            info[t1w] = [s.series_number] # assign if a single series meets criteria
        # if ('gre_field_mapping_2mm' in s.protocol_name):
        #     info[fmap].append(s.series_number)
            
    return info

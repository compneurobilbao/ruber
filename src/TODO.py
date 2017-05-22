#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 22 10:33:07 2017

@author: asier
"""

"""
TODO: Add session_list = ['run001', 'run002'] # list of session identifiers
"""
# Infosource - a function free node to iterate over the list of subject names
infosource = Node(IdentityInterface(fields=['subject_id',
                                            'session_id']),
                  name="infosource")
infosource.iterables = [('subject_id', subject_list),
                        ('session_id', session_list)]

# SelectFiles
templates = {'func': 'data/{subject_id}/{session_id}.nii.gz'}
selectfiles = Node(SelectFiles(templates,
                               base_directory=experiment_dir),
                   name="selectfiles")

# Datasink
datasink = Node(DataSink(base_directory=experiment_dir,
                         container=output_dir),
                name="datasink")

# Use the following DataSink output substitutions
substitutions = [('_subject_id', ''),
                 ('_session_id_', '')]
datasink.inputs.substitutions = substitutions

# Connect SelectFiles and DataSink to the workflow
preproc.connect([(infosource, selectfiles, [('subject_id', 'subject_id'),
                                            ('session_id', 'session_id')]),


"""
TODO: Integrate T1w transformations in pipeline
"""


"""
TODO: Integrate Electrodes T1w in pipeline
"""

"""
TODO: Integrate rsFMRI postproc in pipeline
"""
# -*- coding: utf-8 -*-
"""
Nipype workflows to use Diffusion toolkit for tractography.
"""
from src.env import DATA

import os.path as op
from os.path import join as opj

from nipype.interfaces.io import SelectFiles, DataSink
import nipype.pipeline.engine as pe
import nipype.algorithms.misc as misc
from nipype.interfaces.utility import IdentityInterface

import nipype.interfaces.fsl as fsl          # fsl
import nipype.interfaces.diffusion_toolkit as dtk

from src.utils import (get_datasink,
                       get_interface_node,
                       get_input_node,
                       get_data_dims,
                       get_vox_dims,
                       get_affine,
                       )


def dtk_tractography(wf_name="dtk_tract"):
    """ Run the diffusion MRI pre-processing workflow against the diff files in `data_dir`.

    Nipype Inputs
    -------------
    tract_input.diff: traits.File
        path to the diffusion MRI image

    tract_input.bval: traits.File
        path to the bvals file

    tract_input.bvec: traits.File
        path to the bvecs file

    tract_input.mask: traits.File
        path to the brain mask file

    tract_input.atlas: traits.File
        path to the atlas file

    Nipypte Outputs
    ---------------
    tract_output.tensor
        The result of fitting the tensor model to the whole image.

    tract_output.tracks
        The tractography result.

    tract_output.connectivity
        The atlas ROIxROI structural connectivity matrix.

    tract_output.mean_fa
        The atlas ROIxROI structural connectivity matrix with average FA values.

    tract_output.fa
        The voxelwise fractional anisotropy image.

    Returns
    -------
    wf: nipype Workflow
    """
    in_fields  = ["diff", "bvec", "bval", "mask"]
    out_fields = ["tensor", "tracks"]

    tract_input  = pe.Node(IdentityInterface(fields=in_fields,
                                             mandatory_inputs=True),
                           name="tract_input")

    dtifit = pe.Node(interface=dtk.DTIRecon(),name='dtifit')

    dtk_tracker = pe.Node(interface=dtk.DTITracker(), name="dtk_tracker")
    dtk_tracker.inputs.input_type = "nii.gz"
    smooth_trk = pe.Node(interface=dtk.SplineFilter(), name="smooth_trk")
    smooth_trk.inputs.step_length = 0.5

    tract_output = pe.Node(IdentityInterface(fields=out_fields),
                           name="tract_output")

    # Create the workflow object
    wf = pe.Workflow(name=wf_name)

    # Connect the nodes
    wf.connect([
                # input
                (tract_input, dtifit, [("diff", "DWI")]),
                (tract_input, dtifit, [("bval", "bvals")]),
                (tract_input, dtifit, [("bvec", "bvecs")]),

                # dti fit
                (tract_input, dtk_tracker, [("mask", "mask1_file")]),
                # tracking and smoothing
                (dtifit, dtk_tracker, [("tensor", "tensor_file")]),
                (dtk_tracker, smooth_trk, [('track_file', 'track_file')]),

                # output
                (smooth_trk, tract_output, [("smoothed_track_file", "tracks")]),
                (dtifit, tract_output, [("tensor", "tensor")]),
              ])
    return wf


def run_dtk_tractography(subject_list, session_list):
    """ Attach the Camino-based tractography workflow to the `main_wf`.

    Parameters
    ----------
    main_wf: nipype Workflow

    atlas_file: str
        Path to the anatomical atlas.

    wf_name: str
        Name of the preprocessing workflow

    Nipype Inputs for `main_wf`
    ---------------------------
    Note: The `main_wf` workflow is expected to have an `input_files` and a `datasink` nodes.

    input_files.select.diff: input node

    datasink: nipype Node

    Nipype Workflow Dependencies
    ----------------------------
    This workflow depends on:
    - spm_anat_preproc
    - spm_fsl_dti_preprocessing

    Returns
    -------
    main_wf: nipype Workflow
    """
    output_dir = opj(DATA, 'processed')
    working_dir = opj(DATA, 'interim')


    # Infosource - a function free node to iterate over the list of subject names
    infosource = pe.Node(IdentityInterface(fields=['subject_id',
                                                   'session_id']),
                         name="infosource")
    infosource.iterables = [('subject_id', subject_list),
                            ('session_id', session_list)]

    # SelectFiles
    templates = {'eddy_corr_file': 'processed/diff/_session_id_{session_id}_subject_id_{subject_id}/eddy_corrected_denoised.nii.gz',
                 'bval': 'raw/bids/{subject_id}/{session_id}/dwi/{subject_id}_{session_id}_dwi.bval',
                 'bvec_rotated': 'raw/bids/{subject_id}/{session_id}/dwi/{subject_id}_{session_id}_dwi.bvec',
                 'brain_mask_diff': 'processed/diff/_session_id_{session_id}_subject_id_{subject_id}/eddy_corrected_avg_b0_brain_mask.nii.gz',
                 }
    selectfiles = pe.Node(SelectFiles(templates,
                                      base_directory=DATA),
                          name="selectfiles")
    # Datasink
    datasink = pe.Node(DataSink(base_directory=DATA,
                                container=output_dir),
                       name="datasink")

    # The workflow box
    tract_wf = dtk_tractography()

    wf = pe.Workflow(name='artifact')
    wf.base_dir = working_dir
    # input and output diffusion MRI workflow to main workflow connections
    wf.connect([(infosource, selectfiles, [('subject_id', 'subject_id'),
                                           ('session_id', 'session_id')]),

                (selectfiles, tract_wf, [("bval",            "tract_input.bval"),
                                         ("brain_mask_diff", "tract_input.mask"),
                                         ("eddy_corr_file",  "tract_input.diff"),
                                         ("bvec_rotated",    "tract_input.bvec"),
                                         ]),

                # output
                (tract_wf, datasink, [("tract_output.tensor",       "tract.@tensor_dtk"),
                                      ("tract_output.tracks",       "tract.@tracks_dtk"),
                                      ])
                ])

    # pass the atlas if it's the case
    wf.run()

    return

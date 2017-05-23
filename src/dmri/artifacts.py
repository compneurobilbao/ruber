# -*- coding: utf-8 -*-

"""
Nipype workflow to detect and remove ardifacts in diffusion MRI.
"""
from src.env import DATA

import os.path as op
from os.path import join as opj

from   nipype.interfaces.io import SelectFiles, DataSink
import nipype.pipeline.engine as pe
from   nipype.interfaces.fsl import BET, ExtractROI
from   nipype.interfaces.utility import Function, IdentityInterface
from   nipype.workflows.dmri.fsl.utils import eddy_rotate_bvecs, b0_average, b0_indices
from   nipype.workflows.dmri.fsl import hmc_pipeline

from ..interfaces import Eddy

from .utils import (dti_acquisition_parameters,
                    nlmeans_denoise, reslice,)
#                    rapidart_dti_artifact_detection,)

from .._utils  import format_pair_list
#from ..config  import setup_node, get_config_setting
from ..utils   import (get_datasink,
                       get_input_node,
                       remove_ext,
                       extend_trait_list,
                       get_input_file_name,
                       extension_duplicates,
                       )

def dti_artifact_correction(wf_name="dti_artifact_correction"):
    """ Run the diffusion MRI pre-processing workflow against the diff files in `data_dir`.

    It will resample/regrid the diffusion image to have isometric voxels.
    Corrects for head motion correction and Eddy currents.
    Estimates motion outliers and exports motion reports using nipype.algorithms.RapidArt.

    Nipype Inputs
    -------------
    dti_art_input.diff: traits.File
        path to the diffusion MRI image

    dti_art_input.bval: traits.File
        path to the bvals file

    dti_art_input.bvec: traits.File
        path to the bvecs file


    Nipype Outputs
    --------------
    dti_art_output.eddy_corr_file: traits.File
        Eddy currents corrected DTI image.

    dti_art_output.bvec_rotated: traits.File
        Rotated bvecs file

    dti_art_output.brain_mask_1: traits.File
        Brain mask extracted using BET on the first B0 image.

    dti_art_output.brain_mask_2: traits.File
        Brain mask extracted using BET on the average B0 image,
        after motion correction.

    dti_art_output.acpq: traits.File
        Text file with acquisition parameters calculated for Eddy.

    dti_art_output.index: traits.File
        Text file with acquisition indices calculated for Eddy.

    dti_art_output.avg_b0: traits.File
        The average b=0 image extracted from the motion and eddy
        currents correted diffusion MRI.

    dti_art_output.hmc_corr_file: traits.File

    dti_art_output.hmc_corr_bvec: traits.File

    dti_art_output.hmc_corr_xfms: traits.File

    dti_art_output.art_displacement_files: traits.File

    dti_art_output.art_intensity_files: traits.File

    dti_art_output.art_norm_files: traits.File

    dti_art_output.art_outlier_files: traits.File

    dti_art_output.art_plot_files: traits.File

    dti_art_output.art_statistic_files: traits.File

    Returns
    -------
    wf: nipype Workflow
    """   
    
    
    # specify input and output fields
    in_fields  = ["diff", "bval", "bvec"]
    out_fields = ["eddy_corr_file",
                  "bvec_rotated",
                  "brain_mask_1",
                  "brain_mask_2",
                  "acqp",
                  "index",
                  "avg_b0",
                  "hmc_corr_file",
                  "hmc_corr_bvec",
                  "hmc_corr_xfms",
#                       "art_displacement_files",
#                       "art_intensity_files",
#                       "art_norm_files",
#                       "art_outlier_files",
#                       "art_plot_files",
#                       "art_statistic_files",
                  ]

    # input interface
    dti_input = pe.Node(IdentityInterface(fields=in_fields, mandatory_inputs=True),
                           name="dti_art_input")

    # resample
    resample = pe.Node(Function(function=reslice,
                                   input_names=['in_file', 'new_zooms', 'order', 'out_file'],
                                   output_names=['out_file']),
                          name='dti_reslice')

    ## extract first b0 for Eddy and HMC brain mask
    list_b0 = pe.Node(Function(function=b0_indices,
                               input_names=['in_bval'],
                               output_names=['out_idx'],),
                               name='b0_indices')

    extract_b0 = pe.Node(ExtractROI(t_size=1),
                         name="extract_first_b0")

    # For Eddy, the mask is only used for selecting voxels for the estimation of the hyperparameters,
    # so isn’t very critical.
    # Note also that it is better with a too conservative (small) mask than a too big.
    bet_dwi0 = pe.Node(BET(frac=0.3, mask=True, robust=True),
                          name='bet_dwi_pre')

    pick_first = lambda lst: lst[0]

    # motion artifacts detection, requires linear co-registration for motion estimation.
    
    # head motion correction
    hmc = hmc_pipeline()

    # art = pe.Node(rapidart_dti_artifact_detection(), name="detect_artifacts")

    # Eddy
    eddy = pe.Node(Eddy(method='jac'), name="eddy")

    ## acquisition parameters for Eddy
    write_acqp = pe.Node(Function(function=dti_acquisition_parameters,
                                     input_names=["in_file"],
                                     output_names=["out_acqp", "out_index"],),
                            name="write_acqp")

    ## rotate b-vecs
    rot_bvec = pe.Node(Function(function=eddy_rotate_bvecs,
                                   input_names=["in_bvec", "eddy_params"],
                                   output_names=["out_file"],),
                          name="rot_bvec")

    ## extract all b0s and average them after Eddy correction
    avg_b0_post = pe.Node(Function(function=b0_average,
                                   input_names=['in_dwi', 'in_bval'],
                                   output_names=['out_file'],),
                          name='b0_avg_post')

    bet_dwi1 = pe.Node(BET(frac=0.3, mask=True, robust=True),
                          name='bet_dwi_post')


    
    nlmeans = pe.Node(Function(function=nlmeans_denoise,
                                  input_names=['in_file', 'mask_file', 'out_file', 'N'],
                                  output_names=['out_file']),
                         name='nlmeans_denoise')

    # output interface
    dti_output = pe.Node(IdentityInterface(fields=out_fields),
                            name="dti_art_output")

    # Create the workflow object
    wf = pe.Workflow(name=wf_name)

    # Connect the nodes
    wf.connect([
                # resample to iso-voxel
                (dti_input, resample, [("diff", "in_file"),]),

                # read from input file the acquisition parameters for eddy
                (dti_input, write_acqp, [("diff", "in_file")]),

                # reference mask for hmc and eddy
                (dti_input,  list_b0,    [("bval",     "in_bval")]),
                (resample,   extract_b0, [("out_file", "in_file")]),
                (list_b0,    extract_b0, [(("out_idx", pick_first), "t_min")]),

                (extract_b0, bet_dwi0,   [("roi_file", "in_file")]),

                # Eddy
                (resample,   eddy, [("out_file",  "in_file")]),
                (bet_dwi0,   eddy, [("mask_file", "in_mask")]),
                (dti_input,  eddy, [("bval",      "in_bval"),
                                    ("bvec",      "in_bvec")
                                   ]),
                (write_acqp, eddy, [("out_acqp",  "in_acqp"),
                                    ("out_index", "in_index")
                                   ]),
                
                # non-local means
                (eddy,     nlmeans,   [("out_corrected", "in_file")]),
                (bet_dwi1, nlmeans,   [("mask_file",     "mask_file")]),

                # output
                (nlmeans, dti_output, [("out_file", "eddy_corr_file")]),
               
                # rotate bvecs
                (dti_input, rot_bvec, [("bvec",          "in_bvec")]),
                (eddy,      rot_bvec, [("out_parameter", "eddy_params")]),

                # final avg b0
                (dti_input,   avg_b0_post, [("bval",          "in_bval")]),
                (eddy,        avg_b0_post, [("out_corrected", "in_dwi" )]),
                (avg_b0_post, bet_dwi1,    [("out_file",      "in_file")]),

                # head motion correction
                (dti_input, hmc, [("bval", "inputnode.in_bval"),
                                  ("bvec", "inputnode.in_bvec"),
                                 ]),
                (resample,  hmc, [("out_file",              "inputnode.in_file")]),
                (bet_dwi0,  hmc, [("mask_file",             "inputnode.in_mask")]),
                (list_b0,   hmc, [(("out_idx", pick_first), "inputnode.ref_num"),]),

                # artifact detection
#                    (hmc,      art, [("outputnode.out_file", "realigned_files"),
#                                     ("outputnode.out_xfms", "realignment_parameters"),
#                                    ]),
#                    (bet_dwi1, art, [("mask_file", "mask_file"),]),

                # output
                (hmc, dti_output, [("outputnode.out_file", "hmc_corr_file"),
                                   ("outputnode.out_bvec", "hmc_corr_bvec"),
                                   ("outputnode.out_xfms", "hmc_corr_xfms"),
                                  ]),

#                    (art, dti_output, [("displacement_files",  "art_displacement_files"),
#                                       ("intensity_files",     "art_intensity_files"),
#                                       ("norm_files",          "art_norm_files"),
#                                       ("outlier_files",       "art_outlier_files"),
#                                       ("plot_files",          "art_plot_files"),
#                                       ("statistic_files",     "art_statistic_files"),
#                                      ]),
    
                # output
                (write_acqp,  dti_output,  [("out_acqp",  "acqp"),
                                            ("out_index", "index")]),
                (bet_dwi0,    dti_output,  [("mask_file", "brain_mask_1")]),
                (bet_dwi1,    dti_output,  [("mask_file", "brain_mask_2")]),
                (rot_bvec,    dti_output,  [("out_file",  "bvec_rotated")]),
                (avg_b0_post, dti_output,  [("out_file",  "avg_b0")]),
              ])


    return wf


def run_dti_artifact_correction(subject_list, session_list):
    """ Attach the FSL-based diffusion MRI artifact detection and correction
    workflow to the `main_wf`.

    Parameters
    ----------
    main_wf: nipype Workflow

    wf_name: str
        Name of the preprocessing workflow

    params: dict with parameter values
        atlas_file: str
            Path to the anatomical atlas to be transformed to diffusion MRI space.

    Nipype Inputs for `main_wf`
    ---------------------------
    Note: The `main_wf` workflow is expected to have an `input_files` and a `datasink` nodes.

    input_files.select.diff: input node

    datasink: nipype Node

    Returns
    -------
    main_wf: nipype Workflow
    """
    
    # name of output folder
    output_dir = opj(DATA, 'processed')     
    working_dir = opj(DATA, 'interim') 


    # Infosource - a function free node to iterate over the list of subject names
    infosource = pe.Node(IdentityInterface(fields=['subject_id',
                                                   'session_id']),
                         name="infosource")
    infosource.iterables = [('subject_id', subject_list),
                            ('session_id', session_list)]
    
    # SelectFiles
    templates = {'diff': 'raw/bids/{subject_id}/{session_id}/dwi/{subject_id}_{session_id}_dwi.nii.gz',
                 'bval': 'raw/bids/{subject_id}/{session_id}/dwi/{subject_id}_{session_id}_dwi.bval',
                 'bvec': 'raw/bids/{subject_id}/{session_id}/dwi/{subject_id}_{session_id}_dwi.bvec'}
    selectfiles = pe.Node(SelectFiles(templates,
                                      base_directory=DATA),
                          name="selectfiles")
    
    # Datasink
    datasink = pe.Node(DataSink(base_directory=DATA,
                             container=output_dir),
                    name="datasink")
        
    
    # The workflow box
    art_dti_wf = dti_artifact_correction()

    # dataSink output substitutions
    ## The base name of the 'diff' file for the substitutions
    diff_fbasename = remove_ext(op.basename(get_input_file_name(selectfiles, 'diff')))

    regexp_subst = [
                    (r"/brain_mask_{diff}_space\.nii$", "/brain_mask.nii"),
                    (r"/eddy_corrected\.nii$",          "/{diff}_eddycor.nii"),
                   ]
    regexp_subst = format_pair_list(regexp_subst, diff=diff_fbasename)

    regexp_subst += extension_duplicates(regexp_subst)
    datasink.inputs.regexp_substitutions = extend_trait_list(datasink.inputs.regexp_substitutions,
                                                             regexp_subst)


    wf = pe.Workflow(name='artifact')
    wf.base_dir = working_dir
    
    # input and output diffusion MRI workflow to main workflow connections
    wf.connect([(infosource, selectfiles, [('subject_id', 'subject_id'),
                                           ('session_id', 'session_id')]),
                (selectfiles,   art_dti_wf, [("diff", "dti_art_input.diff"),
                                             ("bval", "dti_art_input.bval"),
                                             ("bvec", "dti_art_input.bvec"),
                                             ]),
                (art_dti_wf, datasink, [("dti_art_output.eddy_corr_file", "diff.@eddy_corr_file"),
                                        ("dti_art_output.bvec_rotated",   "diff.@bvec_rotated"),
                                        ("dti_art_output.brain_mask_1",   "diff.@brain_mask_1"),
                                        ("dti_art_output.brain_mask_2",   "diff.@brain_mask_2"),
                                        ("dti_art_output.acqp",           "diff.@acquisition_pars"),
                                        ("dti_art_output.index",          "diff.@acquisition_idx"),
                                        ("dti_art_output.avg_b0",         "diff.@avg_b0"),
                                        ("dti_art_output.hmc_corr_file",          "diff.artifact_stats.@hmc_corr_file"),
                                        ("dti_art_output.hmc_corr_bvec",          "diff.artifact_stats.@hmc_rot_bvec"),
                                        ("dti_art_output.hmc_corr_xfms",          "diff.artifact_stats.@hmc_corr_xfms"),
#                                                 ("dti_art_output.art_displacement_files", "diff.artifact_stats.@art_disp_files"),
#                                                 ("dti_art_output.art_intensity_files",    "diff.artifact_stats.@art_ints_files"),
#                                                 ("dti_art_output.art_norm_files",         "diff.artifact_stats.@art_norm_files"),
#                                                 ("dti_art_output.art_outlier_files",      "diff.artifact_stats.@art_outliers"),
#                                                 ("dti_art_output.art_plot_files",         "diff.artifact_stats.@art_plots"),
#                                                 ("dti_art_output.art_statistic_files",    "diff.artifact_stats.@art_stats"),
                                    
                                        ]),
                ])


    
    wf.run()
    return

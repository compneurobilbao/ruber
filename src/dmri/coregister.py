# -*- coding: utf-8 -*-
"""
Nipype workflows to co-register anatomical MRI to diffusion MRI.
"""
import nipype.pipeline.engine as pe
from nipype.interfaces.fsl import MultiImageMaths
from nipype.interfaces.utility import IdentityInterface, Select, Split
from nipype.algorithms.misc import Gunzip
from nipype.interfaces.io import SelectFiles, DataSink


from .._utils import flatten_list
from ..preproc import spm_coregister

from os.path import join as opj
import os.path as op


from .artifacts import run_dti_artifact_correction
from .._utils import format_pair_list
# from ..config import check_atlas_file
from ..utils import (get_datasink,
                     get_input_node,
                     get_interface_node,
                     remove_ext,
                     extend_trait_list,
                     get_input_file_name,
                     extension_duplicates,
                     )


def spm_anat_to_diff_coregistration(wf_name="spm_anat_to_diff_coregistration"):
    """ Co-register the anatomical image and other images in anatomical space to
    the average B0 image.

    This estimates an affine transform from anat to diff space, applies it to
    the brain mask and an atlas.

    Nipype Inputs
    -------------
    dti_co_input.avg_b0: traits.File
        path to the average B0 image from the diffusion MRI.
        This image should come from a motion and Eddy currents
        corrected diffusion image.

    dti_co_input.anat: traits.File
        path to the high-contrast anatomical image.

    dti_co_input.tissues: traits.File
        paths to the NewSegment c*.nii output files, in anatomical space

    dti_co_input.atlas_anat: traits.File
        Atlas in subject anatomical space.

    Nipype Outputs
    --------------
    dti_co_output.anat_diff: traits.File
        Anatomical image in diffusion space.

    dti_co_output.tissues_diff: traits.File
        Tissues images in diffusion space.

    dti_co_output.brain_mask_diff: traits.File
        Brain mask for diffusion image.

    dti_co_output.atlas_diff: traits.File
        Atlas image warped to diffusion space.
        If the `atlas_file` option is an existing file and `normalize_atlas` is True.

    Nipype Workflow Dependencies
    ----------------------------
    This workflow depends on:
    - spm_anat_preproc

    Returns
    -------
    wf: nipype Workflow
    """
    # specify input and output fields
    in_fields  = ["avg_b0", "tissues", "anat", "atlas_anat"]
    out_fields = ["anat_diff",
                  "tissues_diff",
                  "brain_mask_diff",
                  "atlas_diff",
                  ]
    
    coreg_atlas = pe.Node(spm_coregister(cost_function="mi"), name="coreg_atlas")
    # set the registration interpolation to nearest neighbour.
    coreg_atlas.inputs.write_interp = 0

    # input interface
    dti_input = pe.Node(IdentityInterface(fields=in_fields, mandatory_inputs=True),
                        name="dti_co_input")

    gunzip_b0 = pe.Node(Gunzip(), name="gunzip_b0")
    coreg_b0  = pe.Node(spm_coregister(cost_function="mi"), name="coreg_b0")

    # co-registration
    brain_sel    = pe.Node(Select(index=[0, 1, 2]),            name="brain_sel")
    coreg_split  = pe.Node(Split(splits=[1, 2], squeeze=True), name="coreg_split")

    brain_merge  = pe.Node(MultiImageMaths(), name="brain_merge")
    brain_merge.inputs.op_string = "-add '%s' -add '%s' -abs -kernel gauss 4 -dilM -ero -kernel gauss 1 -dilM -bin"
    brain_merge.inputs.out_file = "brain_mask_diff.nii.gz"

    # output interface
    dti_output = pe.Node(IdentityInterface(fields=out_fields),
                         name="dti_co_output")

    # Create the workflow object
    wf = pe.Workflow(name=wf_name)

    # Connect the nodes
    wf.connect([
                # co-registration
                (dti_input, coreg_b0, [("anat", "source")]),

                (dti_input,     brain_sel,   [("tissues",             "inlist")]),
                (brain_sel,     coreg_b0,    [(("out", flatten_list), "apply_to_files")]),

                (dti_input,     gunzip_b0,   [("avg_b0",   "in_file")]),
                (gunzip_b0,     coreg_b0,    [("out_file", "target")]),

                (coreg_b0,      coreg_split, [("coregistered_files", "inlist")]),
                (coreg_split,   brain_merge, [("out1",               "in_file")]),
                (coreg_split,   brain_merge, [("out2",               "operand_files")]),

                (dti_input,   coreg_atlas, [("anat",               "source"),
                                                ("atlas_anat",         "apply_to_files"),
                                               ]),
                (gunzip_b0,   coreg_atlas, [("out_file",           "target")]),
                (coreg_atlas, dti_output,  [("coregistered_files", "atlas_diff")]),
              
                # output
                (coreg_b0,     dti_output,     [("coregistered_source", "anat_diff")]),
                (coreg_b0,     dti_output,     [("coregistered_files",  "tissues_diff")]),
                (brain_merge,  dti_output,     [("out_file",            "brain_mask_diff")]),
              ])

    return wf


def run_spm_fsl_dti_preprocessing(experiment_dir, subject_list):
    """ Attach a set of pipelines to the `main_wf` for Diffusion MR (`diff`) image processing.
    - dti_artifact_correction
    - spm_anat_to_diff_coregistration
    - dti_tensor_fitting

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
    output_dir = opj(experiment_dir, 'data', 'processed')     
    working_dir = opj(experiment_dir,'data', 'interim') 


    # Infosource - a function free node to iterate over the list of subject names
    infosource = pe.Node(IdentityInterface(fields=['subject_id']),
                      name="infosource")
    infosource.iterables = [('subject_id', subject_list)]
    
    # SelectFiles
    templates = {'avg_b0': 'data/processed/diff/_subject_id_{subject_id}/eddy_corrected_avg_b0.nii.gz',
                 'tissues_native': 'data/processed/fmriprep/{subject_id}/anat/{subject_id}_T1w_dtissue.nii.gz',
                 'anat_biascorr': 'data/processed/fmriprep/{subject_id}/anat/{subject_id}_T1w_preproc.nii.gz',
                 'atlas_anat': 'data/processed/fmriprep/{subject_id}/anat/{subject_id}__atlas.nii.gz',
                 }
    selectfiles = pe.Node(SelectFiles(templates,
                                      base_directory=experiment_dir),
                          name="selectfiles")
    
    # Datasink
    datasink = pe.Node(DataSink(base_directory=experiment_dir,
                             container=output_dir),
                    name="datasink")
        

    # The workflow boxes
    coreg_dti_wf = spm_anat_to_diff_coregistration()

    # dataSink output substitutions
    ## The base name of the 'diff' file for the substitutions
    diff_fbasename = remove_ext(op.basename(get_input_file_name(selectfiles, 'diff')))
    anat_fbasename = remove_ext(op.basename(get_input_file_name(selectfiles, 'anat')))

    regexp_subst = [
                    (r"/brain_mask_{diff}_space\.nii$", "/brain_mask.nii"),
                    (r"/eddy_corrected\.nii$",          "/{diff}_eddycor.nii"),
                    (r"/rc1anat_hc_corrected\.nii$",    "/gm_diff.nii"),
                    (r"/rc2anat_hc_corrected\.nii$",    "/wm_diff.nii"),
                    (r"/rc3anat_hc_corrected\.nii$",    "/csf_diff.nii"),
                    (r"/rmanat_hc_corrected\.nii$",     "/{anat}_diff.nii"),
                   ]
    regexp_subst = format_pair_list(regexp_subst, diff=diff_fbasename,
                                                  anat=anat_fbasename)

    # prepare substitution for atlas_file

    atlas_basename = remove_ext(op.basename(get_input_file_name(selectfiles, 'atlas_anat')))
    regexp_subst.extend([
                         (r"/[\w]*{atlas}.*\.nii$", "/{atlas}_{diff}_space.nii"),
                        ])
    regexp_subst = format_pair_list(regexp_subst, atlas=atlas_basename,
                                                  diff=diff_fbasename)


    regexp_subst += extension_duplicates(regexp_subst)
    datasink.inputs.regexp_substitutions = extend_trait_list(datasink.inputs.regexp_substitutions,
                                                             regexp_subst)

    wf = pe.Workflow(name='artifact')
    wf.base_dir = working_dir
    
    # input and output diffusion MRI workflow to main workflow connections
    wf.connect([(infosource, selectfiles, [('subject_id', 'subject_id')]),
                (selectfiles, coreg_dti_wf, [("avg_b0",         "dti_co_input.avg_b0"),]),
                (selectfiles, coreg_dti_wf, [("tissues_native", "dti_co_input.tissues"),
                                             ("anat_biascorr",  "dti_co_input.anat")
                                            ]),
                (selectfiles,  coreg_dti_wf, [("atlas_anat", "dti_co_input.atlas_anat")]),
                (coreg_dti_wf, datasink,     [("dti_co_output.atlas_diff", "diff.@atlas")]),
                (coreg_dti_wf, datasink, [("dti_co_output.anat_diff",       "diff.@anat_diff"),
                                           ("dti_co_output.tissues_diff",    "diff.tissues.@tissues_diff"),
                                           ("dti_co_output.brain_mask_diff", "diff.@brain_mask"),
                                          ]),
                ])
    
    wf.run()
    return main_wf

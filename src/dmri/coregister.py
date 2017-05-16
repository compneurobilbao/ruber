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
    in_fields  = ["avg_b0", "brain_mask", "anat", "atlas_2514", "atlas_2754"]
    out_fields = ["anat_diff",
                  "brain_mask_diff",
                  "atlas_2514_diff",
                  "atlas_2754_diff",
                  ]
    
    gunzip_atlas_2514 = pe.Node(Gunzip(), name="gunzip_atlas_2514")
    gunzip_atlas_2754 = pe.Node(Gunzip(), name="gunzip_atlas_2754")
    gunzip_anat = pe.Node(Gunzip(), name="gunzip_anat")
    gunzip_brain_mask = pe.Node(Gunzip(), name="brain_mask")
    coreg_atlas_2514 = pe.Node(spm_coregister(cost_function="mi"), name="coreg_atlas_2514")
    # set the registration interpolation to nearest neighbour.
    coreg_atlas_2514.inputs.write_interp = 0
    
    coreg_atlas_2754 = pe.Node(spm_coregister(cost_function="mi"), name="coreg_atlas_2754")
    # set the registration interpolation to nearest neighbour.
    coreg_atlas_2754.inputs.write_interp = 0

    # input interface
    dti_input = pe.Node(IdentityInterface(fields=in_fields, mandatory_inputs=True),
                        name="dti_co_input")

    gunzip_b0 = pe.Node(Gunzip(), name="gunzip_b0")
    coreg_b0  = pe.Node(spm_coregister(cost_function="mi"), name="coreg_b0")

    # co-registration
    coreg_brain  = pe.Node(spm_coregister(cost_function="mi"), name="coreg_brain")
    # set the registration interpolation to nearest neighbour.
    coreg_brain.inputs.write_interp = 0
    
    
    # output interface
    dti_output = pe.Node(IdentityInterface(fields=out_fields),
                         name="dti_co_output")

    # Create the workflow object
    wf = pe.Workflow(name=wf_name)

    # Connect the nodes
    wf.connect([(dti_input, gunzip_atlas_2514,   [("atlas_2514",   "in_file")]),
                (dti_input, gunzip_atlas_2754,   [("atlas_2754",   "in_file")]),
                (dti_input,  gunzip_anat , [("anat",          "in_file")]),
                (dti_input,     gunzip_b0,   [("avg_b0",   "in_file")]),
                (dti_input,     gunzip_brain_mask,   [("brain_mask",   "in_file")]),
                # co-registration
                
                
                # some of this code is not needed
                (gunzip_b0,  coreg_b0,    [("out_file", "target")]),
                (gunzip_brain_mask, coreg_b0, [("out_file",   "apply_to_files")]),
                (gunzip_anat, coreg_b0, [("out_file", "source")]),

                (gunzip_b0,     coreg_atlas_2514,    [("out_file", "target")]),
                (gunzip_atlas_2514,   coreg_atlas_2514, [("out_file",   "apply_to_files")]),
                (gunzip_anat,   coreg_atlas_2514, [("out_file",    "source"), ]),
                
                (gunzip_b0,     coreg_atlas_2754,    [("out_file", "target")]),
                (gunzip_atlas_2754,   coreg_atlas_2754, [("out_file",   "apply_to_files")]),
                (gunzip_anat,   coreg_atlas_2754, [("out_file",    "source"), ]),
                
                (gunzip_b0,     coreg_brain,    [("out_file", "target")]),
                (gunzip_brain_mask,   coreg_brain, [("out_file",   "apply_to_files")]),
                (gunzip_anat,   coreg_brain, [("out_file",    "source"), ]),
                
                # output
                (coreg_atlas_2514, dti_output,  [("coregistered_files", "atlas_2514_diff")]),
                (coreg_atlas_2754, dti_output,  [("coregistered_files", "atlas_2754_diff")]),
                (coreg_b0,     dti_output,     [("coregistered_source", "anat_diff")]),
                (coreg_brain,  dti_output,     [("coregistered_files",  "brain_mask_diff")]),
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
                 'brain_mask': 'data/processed/fmriprep/{subject_id}/anat/{subject_id}_T1w_brainmask.nii.gz',
                 'anat_biascorr': 'data/processed/fmriprep/{subject_id}/anat/{subject_id}_T1w_preproc.nii.gz',
                 'atlas_2514': 'data/processed/fmriprep/{subject_id}/anat/{subject_id}_atlas_2514.nii.gz',
                 'atlas_2754': 'data/processed/fmriprep/{subject_id}/anat/{subject_id}_atlas_2754.nii.gz',
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
#    diff_fbasename = remove_ext(op.basename(get_input_file_name(selectfiles, 'avg_b0')))
#    anat_fbasename = remove_ext(op.basename(get_input_file_name(selectfiles, 'anat_biascorr')))
#
#    regexp_subst = [
#                    (r"/brain_mask_{diff}_space\.nii$", "/brain_mask.nii"),
#                    (r"/eddy_corrected\.nii$",          "/{diff}_eddycor.nii"),
#                    (r"/rc1anat_hc_corrected\.nii$",    "/gm_diff.nii"),
#                    (r"/rc2anat_hc_corrected\.nii$",    "/wm_diff.nii"),
#                    (r"/rc3anat_hc_corrected\.nii$",    "/csf_diff.nii"),
#                    (r"/rmanat_hc_corrected\.nii$",     "/{anat}_diff.nii"),
#                   ]
#    regexp_subst = format_pair_list(regexp_subst, diff=diff_fbasename,
#                                                  anat=anat_fbasename)
#
#    # prepare substitution for atlas_file
#
#    atlas_basename = remove_ext(op.basename(get_input_file_name(selectfiles, 'atlas_anat')))
#    regexp_subst.extend([
#                         (r"/[\w]*{atlas}.*\.nii$", "/{atlas}_{diff}_space.nii"),
#                        ])
#    regexp_subst = format_pair_list(regexp_subst, atlas=atlas_basename,
#                                                  diff=diff_fbasename)
#
#
#    regexp_subst += extension_duplicates(regexp_subst)
#    datasink.inputs.regexp_substitutions = extend_trait_list(datasink.inputs.regexp_substitutions,
#                                                             regexp_subst)

    wf = pe.Workflow(name='artifact')
    wf.base_dir = working_dir
    
    # input and output diffusion MRI workflow to main workflow connections
    wf.connect([(infosource, selectfiles, [('subject_id', 'subject_id')]),
                (selectfiles, coreg_dti_wf, [("avg_b0",         "dti_co_input.avg_b0"),]),
                (selectfiles, coreg_dti_wf, [("brain_mask", "dti_co_input.brain_mask"),
                                             ("anat_biascorr",  "dti_co_input.anat")
                                            ]),
                (selectfiles,  coreg_dti_wf, [("atlas_2514", "dti_co_input.atlas_2514")]),
                (selectfiles,  coreg_dti_wf, [("atlas_2754", "dti_co_input.atlas_2754")]),
                (coreg_dti_wf, datasink,     [("dti_co_output.atlas_2514_diff", "diff.@atlas_2514")]),
                (coreg_dti_wf, datasink,     [("dti_co_output.atlas_2754_diff", "diff.@atlas_2754")]),
                (coreg_dti_wf, datasink, [("dti_co_output.anat_diff",       "diff.@anat_diff"),
                                           ("dti_co_output.brain_mask_diff", "diff.@brain_mask"),
                                          ]),
                ])
    
    wf.run()
    return 

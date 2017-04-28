# -*- coding: utf-8 -*-
"""
Nipype workflows to use Camino for tractography.
"""
import os.path as op
from os.path import join as opj

from   nipype.interfaces.io import SelectFiles, DataSink
import nipype.pipeline.engine    as pe
import nipype.algorithms.misc    as misc
from   nipype.interfaces.utility import IdentityInterface
from   nipype.interfaces.camino  import (Image2Voxel, FSL2Scheme, DTIFit, Track,
                                         Conmat, ComputeFractionalAnisotropy, AnalyzeHeader)

#from   src.config  import setup_node, check_atlas_file
from   src.utils import (get_datasink,
                       get_interface_node,
                       get_input_node,
                       get_data_dims,
                       get_vox_dims,
                       get_affine,
                       )


def camino_tractography(wf_name="camino_tract"):
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
    in_fields  = ["diff", "bvec", "bval", "mask", "atlas"]
    out_fields = ["tensor", "tracks", "connectivity", "mean_fa", "fa"]

    tract_input  = pe.Node(IdentityInterface(fields=in_fields,
                                             mandatory_inputs=True),
                           name="tract_input")

    img2vox_diff = pe.Node(Image2Voxel(out_type="float"), name="img2vox_diff")
    img2vox_mask = pe.Node(Image2Voxel(out_type="short"), name="img2vox_mask")
    fsl2scheme   = pe.Node(FSL2Scheme(),                  name="fsl2scheme")
    dtifit       = pe.Node(DTIFit(),                      name="dtifit")
    fa           = pe.Node(ComputeFractionalAnisotropy(), name="fa")

    analyzehdr_fa = pe.Node(interface=AnalyzeHeader(), name="analyzeheader_fa")
    analyzehdr_fa.inputs.datatype = "double"

    fa2nii = pe.Node(interface=misc.CreateNifti(), name='fa2nii')

    track  = pe.Node(Track(inputmodel="dt", out_file="tracts.Bfloat"), name="track")
    conmat = pe.Node(Conmat(output_root="conmat_"), name="conmat")

    tract_output = pe.Node(IdentityInterface(fields=out_fields),
                           name="tract_output")

    # Create the workflow object
    wf = pe.Workflow(name=wf_name)

    # Connect the nodes
    wf.connect([
                # convert data to camino format
                (tract_input,   img2vox_diff,     [("diff",                  "in_file"     )]),
                (tract_input,   img2vox_mask,     [("mask",                  "in_file"     )]),

                # convert bvec and bval to camino scheme
                (tract_input,   fsl2scheme,       [("bvec",                  "bvec_file"   ),
                                                   ("bval",                  "bval_file"   )]),

                # dtifit
                (img2vox_diff,  dtifit,           [("voxel_order",           "in_file"     )]),
                (img2vox_mask,  dtifit,           [("voxel_order",           "bgmask"      )]),
                (fsl2scheme,    dtifit,           [("scheme",                "scheme_file" )]),

                # calculate FA
                (fsl2scheme,    fa,               [("scheme",                "scheme_file" )]),
                (dtifit,        fa,               [("tensor_fitted",         "in_file"     )]),

                # tractography
                (tract_input,   track,            [("atlas",                 "seed_file"   )]),
                (dtifit,        track,            [("tensor_fitted",         "in_file"     )]),

                # convert FA data to NifTI
                (fa,            analyzehdr_fa,    [("fa",                    "in_file"     )]),
                (tract_input,   analyzehdr_fa,    [(('diff', get_vox_dims),  "voxel_dims"  ),
                                                   (('diff', get_data_dims), "data_dims"   )]),

                (tract_input,   fa2nii,           [(("diff", get_affine),    "affine"      )]),
                (analyzehdr_fa, fa2nii,           [("header",                "header_file" )]),
                (fa,            fa2nii,           [("fa",                    "data_file"   )]),

                # connectivity matrix
                (tract_input,   conmat,           [("atlas",                 "target_file" )]),
                (track,         conmat,           [("tracked",               "in_file"     )]),
                (fa2nii,        conmat,           [("nifti_file",            "scalar_file" )]),

                # output
                (fa2nii,        tract_output,     [("nifti_file",            "fa"          )]),
                (dtifit,        tract_output,     [("tensor_fitted",         "tensor"      )]),
                (track,         tract_output,     [("tracked",               "tracks"      )]),
                (conmat,        tract_output,     [("conmat_sc",             "connectivity"),
                                                   ("conmat_ts",             "mean_fa"     )]),
              ])
    return wf


def run_camino_tractography(experiment_dir, subject_list):
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
    output_dir = opj(experiment_dir, 'data', 'processed')     
    working_dir = opj(experiment_dir,'data', 'interim') 


    # Infosource - a function free node to iterate over the list of subject names
    infosource = pe.Node(IdentityInterface(fields=['subject_id']),
                      name="infosource")
    infosource.iterables = [('subject_id', subject_list)]
    
    # SelectFiles
    templates = {'eddy_corr_file': 'data/processed/diff/_subject_id_{subject_id}/eddy_corrected_denoised.nii.gz',
                 'bval': 'data/raw/bids/{subject_id}/dwi/{subject_id}_dwi.bval',
                 'bvec_rotated': 'data/processed/diff/_subject_id_{subject_id}/{subject_id}_dwi_rotated.bvec',
                 # TODO: Check if correct mask
                 'brain_mask_diff': 'data/processed/diff/_subject_id_{subject_id}/r{subject_id}_T1w_brainmask.nii',
                 'atlas_diff': 'data/processed/diff/_subject_id_{subject_id}/r{subject_id}_atlas.nii'}
    selectfiles = pe.Node(SelectFiles(templates,
                                      base_directory=experiment_dir),
                          name="selectfiles")
    # Datasink
    datasink = pe.Node(DataSink(base_directory=experiment_dir,
                             container=output_dir),
                    name="datasink")


    # The workflow box
    tract_wf = camino_tractography()

    wf = pe.Workflow(name='artifact')
    wf.base_dir = working_dir
    # input and output diffusion MRI workflow to main workflow connections
    wf.connect([(infosource, selectfiles, [('subject_id', 'subject_id')]),
                
                (selectfiles, tract_wf, [("bval",            "tract_input.bval"),
                                         ("brain_mask_diff", "tract_input.mask"),
                                         ("eddy_corr_file",  "tract_input.diff"),
                                         ("bvec_rotated",    "tract_input.bvec"),
                                         ("atlas_diff", "tract_input.atlas")]),

                # output
                (tract_wf, datasink, [("tract_output.tensor",       "tract.@tensor"),
                                      ("tract_output.tracks",       "tract.@tracks"),
                                      ("tract_output.connectivity", "tract.@connectivity"),
                                      ("tract_output.mean_fa",      "tract.@mean_fa"),
                                      ("tract_output.fa",           "tract.@fa"),
                                      ])
                ])

    # pass the atlas if it's the case
    wf.run()
    
    return

# -*- coding: utf-8 -*-
"""
Utilities to help in the DTI pre-processing
"""
from nipype.algorithms.rapidart import ArtifactDetect
from src.env import DATA, ATLAS_TYPES
from src.postproc.utils import execute
import os
from os.path import join as opj
import nibabel as nib
import numpy as np


def nlmeans_denoise(in_file, mask_file, out_file='', N=12):
    """ Filepath interface to the nlmeans_denoise_img in pypes.preproc."""
    import os.path as op
    import nibabel as nib

    from src.preproc import nlmeans_denoise_img
    from src.utils import rename

    den = nlmeans_denoise_img(nib.load(in_file), mask=nib.load(mask_file), N=N)

    if not out_file:
        out_file = rename(in_file, '_denoised')

    den.to_filename(out_file)

    return op.abspath(out_file)


def reslice(in_file, new_zooms=None, order=3, out_file=''):
    """
    Performs regridding of an image to set isotropic voxel sizes using dipy.
    """
    import os.path as op

    import nibabel as nib

    from src.preproc import reslice_img
    from src.utils import rename

    img = reslice_img(nib.load(in_file), new_zooms=new_zooms, order=order)

    if not out_file:
        out_file = rename(in_file, '_resliced')

    img.to_filename(out_file)

    return op.abspath(out_file)


def dti_acquisition_parameters(in_file, epi_factor=128):
    """ Return the path to the acqp_file and index file necessary for the new Eddy tool.
    This works when `in_file` is a nifti file obtained from a recent version from dcm2nii.

    This assumes a standard Siemens acquisition if the 'descrip.phaseDir' field is not
    in the nifti header.

    Please check the InPlanePhaseEncodingDirection field in the DICOM files, if its values is
    'COL', you are good to go.

    Notes
    -----
    # Comments on the `eddy` tool from FSL FDT.

    A description of the tool:
    http://fsl.fmrib.ox.ac.uk/fsl/fslwiki/Eddy/UsersGuide

    Our problem to run this tool instead of the good-old `eddy_correct` is the `--acqp` argument, an
    acquisitions parameters file.

    A detailed description of the --acpq input file is here:
    http://fsl.fmrib.ox.ac.uk/fsl/fslwiki/eddy/Faq#How_do_I_know_what_to_put_into_my_--acqp_file

    In the following subsections I describe each of the fields needed to check and build the acquisitions parameters
    file.

    ## Phase Encoding Direction
    Dicom header field: (0018,1312) InPlanePhaseEncodingDirection
    The phase encoding direction is the OPPOSITE of frequency encoding direction:
    - 'COL' = A/P (freqDir L/R),
    - 'ROW' = L/R (freqDir A/P)


    Nifti header field: "descrip.phaseDir:'+'" is for 'COL' in the DICOM InPlanePhaseEncodingDirection value.
    So if you have only one phase encoding oriendation and a '+' in the previous header field,
    the first 3 columns for the `--acqp` parameter file should be:
    0 1 0

    indicating that the scan was acquired with phase-encoding in the anterior-posterior direction.

    For more info:
    http://web.stanford.edu/group/vista/cgi-bin/wiki/index.php/DTI_Preprocessing_User_Manual#Frequency_vs_Phase_Encode_Direction
    https://xwiki.nbirn.org:8443/xwiki/bin/view/Function-BIRN/PhaseEncodeDirectionIssues


    ## Effective Echo Spacing (aka dwell time)

    Effective Echo Spacing (s) = 1/(BandwidthPerPixelPhaseEncode * MatrixSizePhase)

    effective echo spacing = 1 / [(0019,1028) * (0051,100b component #1)] (from the archives)
    https://www.jiscmail.ac.uk/cgi-bin/webadmin?A3=ind1303&L=FSL&E=quoted-printable&P=29358&B=--B_3444933351_15386849&T=text%2Fhtml;%20charset=ISO-8859-1&pending=

    The dwell time is in the nifti header `descrip.dwell`, in seconds (or look at the field `time_units`).
    http://www.mit.edu/~satra/nipype-nightly/interfaces/generated/nipype.interfaces.fsl.epi.html

    More info:
    http://lcni.uoregon.edu/kb-articles/kb-0003

    ## EPI factor

    The EPI factor is not included in the nifti header.
    You can read it using the Grassroots DICOM tool called `gdcmdump`, for example:
    >>> gdcmdump -C IM-0126-0001.dcm | grep 'EPIFactor'
    sFastImaging.lEPIFactor                  = 128

    More info:
    http://dicomlookup.com/default.htm


    # The fourth element of the acquisitions parameter file

    The fourth element in each row is the time (in seconds) between reading the center of the first echo and reading the
    center of the last echo.
    It is the "dwell time" multiplied by "number of PE steps - 1" and it is also the reciprocal of the PE bandwidth/pixel.

    Total readout time (FSL) = (number of echoes - 1) * echo spacing

    Total Readout Time (SPM) = 1/(BandwidthPerPixelPhaseEncode)
    Since the Bandwidth Per Pixel Phase Encode is in Hz, this will give the readout time in seconds


    # The `---index` argument

    The index argument is a text file with a row of numbers. Each number
    indicates what line (starting from 1) in the `acqp` file corresponds to
    each volume in the DTI acquisition.

    # What to do now with the `dcm2nii` files?

    I see two options to calculate the `acqp` lines with these files.

    1. We already have the `dwell` but we don't have the EPI factor.
    We know that the standard in Siemens is 128 and we could stick to that.

    2. Use the `slice_duration * 0.001` which is very near the calculated value.

    # Summary

    So, for example, if we had these acquisition parameters:

    ```
    Phase enc. dir. P >> A
    Echo spacing 0.75 [ms]
    EPI factor 128
    ```

    We should put in the `acqp` file this line:
    0 1 0 0.095
    """
    # the imports must be inside if you want them to work in a nipype.Function node.
    import warnings
    import os.path as op
    import nibabel as nib

    acqp_file = "diff.acqp"
    index_file = "diff.index"

    image = nib.load(in_file)
    n_volumes = image.shape[-1]

    # read the value of the descrip field in the nifti header
    descrip_field = image.header["descrip"].astype(str)[()]
    descrip = dict([item.split("=", 1) for item in descrip_field.split(";")])

    # assume by default phaseDir = '+'
    phase_dir = descrip.get("phase", '+')
    pe_axis   = "0 1 0"
    if phase_dir == "-":
        pe_axis = "0 -1 0"

    if 'phase' not in descrip:
        warnings.warn("Could not find or understand the value for phaseDir: {}. "
                      "Using default PE axis {}.".format(descrip.get("phase"), pe_axis),
                      category=RuntimeWarning, stacklevel=2)

    # (number of phase-encode steps - 1) *
    # (echo spacing time in milliseconds) *
    # (seconds per millisecond)
    total_readout_time = (epi_factor - 1) * float(descrip["dwell"]) * 1e-3

    with open(acqp_file, "wt") as fout:
        fout.write("{} {}\n".format(pe_axis, total_readout_time))

    with open(index_file, "wt") as fout2:
        fout2.write("{}\n".format(" ".join(n_volumes * ["1"])))

    return op.abspath(acqp_file), op.abspath(index_file)


def correct_dwi_space_atlas(subject_list, session_list):
    """
    Function to correct atlas after resampling (looses some rois),
    this function recovers those lost rois
    """
    sub_ses_comb = [[subject, session] for subject in subject_list
                    for session in session_list]

    for sub, ses in sub_ses_comb:
        for atlas in ATLAS_TYPES:

            atlas_old = opj(DATA, 'external',
                            'bha_' + atlas + '_1mm_mni09c.nii.gz')
            atlas_new = opj(DATA, 'processed', 'diff', '_session_id_' +
                            ses + '_subject_id_' + sub,
                            'r' + sub + '_' + ses + '_' + atlas + '.nii')

            atlas_new_img = nib.load(atlas_new)
            m = atlas_new_img.affine[:3, :3]

            atlas_old_data = nib.load(atlas_old).get_data()
            atlas_old_data_rois = np.unique(atlas_old_data)
            atlas_new_data = atlas_new_img.get_data()
            atlas_new_data_rois = np.unique(atlas_new_data)

            diff_rois = np.setdiff1d(atlas_old_data_rois, atlas_new_data_rois)

            for roi in diff_rois:
                p = np.argwhere(atlas_old_data == roi)[0]
                x, y, z = (np.round(np.diag(np.divide(p, m)))).astype(int)
                atlas_new_data[x, y, z] = roi

            atlas_new_data_img_corrected = nib.Nifti1Image(atlas_new_data,
                                                           affine=atlas_new_img.affine)
            os.remove(atlas_new)
            nib.save(atlas_new_data_img_corrected,
                     atlas_new)


def compress_file_gz(path_file):
    from subprocess import check_call

    check_call(['gzip', path_file])
    check_call(['rm', path_file])

    return path_file + '.gz'


def get_con_matrix_matlab(subject_list, session_list):
    """
    Function to get atlas-wise connectivity matrix. End-to-end fiber counting.
    Camino's workflow is not calculating end-to-end
    """
    sub_ses_comb = [[subject, session] for subject in subject_list
                    for session in session_list]

    for sub, ses in sub_ses_comb:
        for atlas in ATLAS_TYPES:

            num_nodes = atlas[-4:]
            out_folder = opj(DATA, 'processed', 'tract',
                             '_session_id_' + ses + '_subject_id_' + sub)
            tracts_file = opj(out_folder, 'tracts.Bfloat_' + num_nodes)
            reference_file = opj(DATA, 'processed', 'diff',
                                 '_session_id_' + ses + '_subject_id_' + sub,
                                 'eddy_corrected_avg_b0.nii.gz')
            atlas_file = opj(DATA, 'processed', 'diff',
                             '_session_id_' + ses + '_subject_id_' + sub,
                             'r' + sub + '_' + ses + '_' + atlas + '.nii')

            try:
                atlas_file = compress_file_gz(atlas_file)
            except:
                atlas_file = atlas_file + '.gz'

            matlab_path = opj(os.getcwd(), 'src', 'matlab')

            # Extract brain from subject space
            command = ["matlab",
                       "-nodisplay",
                       "-nojvm",
                       "-r",
                       "addpath(\'" + matlab_path + "\');" +
                       "calc_cm(" +
                       "\'" + tracts_file + "\'," +
                       "\'" + reference_file + "\'," +
                       "\'" + atlas_file + "\'," +
                       "\'" + out_folder + "\'," +
                       num_nodes +
                       "); exit;"
                       ]
            for output in execute(command):
                print(output)

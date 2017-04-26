# -*- coding: utf-8 -*-

#extract_b0.max_b: 0.0 # maximum b-value to be considered a b0, for brain mask calculation.
#dmri.artifact_detect: True # to compute motion statistics for post-hoc outliers exclusion.
#dmri.apply_nlmeans: True # to apply non-local means to the corrected DTI data.
#
#coreg_b0.write_interp: 3 # degree of b-spline used for interpolation
#nlmeans_denoise.N: 12 # number of channels in the head coil
#
## Camino Tractography
#conmat.tract_stat: "mean"
#track.curvethresh: 50
#track.anisthresh: 0.2


from os.path import join as opj

from src.dmri import attach_spm_fsl_dti_preprocessing
from src.dmri import attach_camino_tractography  

from nipype.interfaces.spm import SliceTiming, Realign, Smooth
from nipype.interfaces.utility import IdentityInterface
from nipype.interfaces.io import SelectFiles, DataSink
from nipype.algorithms.rapidart import ArtifactDetect
from nipype.algorithms.misc import Gunzip
from nipype.pipeline.engine import Workflow, Node


experiment_dir = '~/nipype_tutorial'             # location of experiment folder
data_dir = opj(experiment_dir, 'data')  # location of data folder
fs_folder = opj(experiment_dir, 'freesurfer')  # location of freesurfer folder

subject_list = ['sub001', 'sub002', 'sub003']    # list of subject identifiers
session_list = ['run001', 'run002']              # list of session identifiers

output_dir = 'output_firstSteps'          # name of output folder
working_dir = 'workingdir_firstSteps'     # name of working directory

number_of_slices = 40                     # number of slices in volume
TR = 2.0                                  # time repetition of volume
smoothing_size = 8    

# Gunzip - unzip functional
gunzip = Node(Gunzip(), name="gunzip")

# Slicetiming - correct for slice wise acquisition
interleaved_order = range(1,number_of_slices+1,2) + range(2,number_of_slices+1,2)
sliceTiming = Node(SliceTiming(num_slices=number_of_slices,
                               time_repetition=TR,
                               time_acquisition=TR-TR/number_of_slices,
                               slice_order=interleaved_order,
                               ref_slice=2),
                   name="sliceTiming")

# Realign - correct for motion
realign = Node(Realign(register_to_mean=True),
               name="realign")

# Artifact Detection - determine which of the images in the functional series
#   are outliers. This is based on deviation in intensity or movement.
art = Node(ArtifactDetect(norm_threshold=1,
                          zintensity_threshold=3,
                          mask_type='spm_global',
                          parameter_source='SPM'),
           name="art")

# Smooth - to smooth the images with a given kernel
smooth = Node(Smooth(fwhm=smoothing_size),
              name="smooth")

# Create a preprocessing workflow
preproc = Workflow(name='preproc')
preproc.base_dir = opj(experiment_dir, working_dir)

# Connect all components of the preprocessing workflow
preproc.connect([(gunzip, sliceTiming, [('out_file', 'in_files')]),
                 (sliceTiming, realign, [('timecorrected_files', 'in_files')]),
                 (realign, art, [('realigned_files', 'realigned_files'),
                                 ('mean_image', 'mask_file'),
                                 ('realignment_parameters',
                                  'realignment_parameters')]),
                 (realign, smooth, [('realigned_files', 'in_files')]),
                 ])


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
                 (selectfiles, gunzip, [('func', 'in_file')]),
                 (realign, datasink, [('mean_image', 'realign.@mean'),
                                      ('realignment_parameters',
                                       'realign.@parameters'),
                                      ]),
                 (smooth, datasink, [('smoothed_files', 'smooth')]),
                 (art, datasink, [('outlier_files', 'art.@outliers'),
                                  ('plot_files', 'art.@plot'),
                                  ]),
                 ])


preproc.write_graph(graph2use='flat')
preproc.run('MultiProc', plugin_args={'n_procs': 6})
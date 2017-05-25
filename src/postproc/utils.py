#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 17 11:23:57 2017

@author: asier
"""
from src.env import DATA, ATLAS_TYPES, NEIGHBOURS

import os.path as op
from os.path import join as opj
import numpy as np
import nibabel as nib
import subprocess

PROCESSED = opj(DATA, 'processed', 'fmriprep')
EXTERNAL = opj(DATA, 'external')
EXTERNAL_MNI_09c = opj(EXTERNAL, 'standard_mni_asym_09c')


def execute(cmd):
    popen = subprocess.Popen(cmd,
                             stdout=subprocess.PIPE,
                             universal_newlines=True)
    for stdout_line in iter(popen.stdout.readline, ""):
        yield stdout_line
    popen.stdout.close()
    return_code = popen.wait()
    if return_code:
        raise subprocess.CalledProcessError(return_code, cmd)


def scrubbing(time_series, FD, thres=0.2):
    """
    simple scrubbing strategy based on timepoint removal
    """
    scrubbed_time_series = time_series.T[:, (FD < thres)]
    return scrubbed_time_series.T


def atlas_to_t1(subject_list, session_list):
    """
    Atlas to T1w space
    """
    sub_ses_comb = [[subject, session] for subject in subject_list
                    for session in session_list]

    for sub, ses in sub_ses_comb:
        # TODO: CORRECT if exists
        #  not op.exists(op.join(PROCESSED, 'fmriprep', 'sub-' + sub,
#                                 'ses-' + ses))
        if True:
            print('Calculating: Subject ', sub, ' and session', ses)

            # Extract brain from subject space
            command = ['fslmaths',
                       opj(PROCESSED, sub, ses, 'anat',
                           sub + '_' + ses + '_T1w_preproc.nii.gz'),
                       '-mas',
                       opj(PROCESSED, sub, ses, 'anat',
                           sub + '_' + ses + '_T1w_brainmask.nii.gz'),
                       opj(PROCESSED, sub, ses, 'anat',
                           sub + '_' + ses + '_T1w_brain.nii.gz'),
                       ]
            for output in execute(command):
                print(output)

            # Brain 09c -> Brain subject (save omat)
            command = ['flirt',
                       '-in',
                       opj(EXTERNAL_MNI_09c,
                           'mni_icbm152_t1_tal_nlin_asym_09c_brain.nii'),
                       '-ref',
                       opj(PROCESSED, sub, ses, 'anat',
                           sub + '_' + ses + '_T1w_brain.nii.gz'),
                       '-omat',
                       opj(PROCESSED, sub, ses, 'anat',
                           '09c_2_' + sub + '_' + ses + '.mat'),
                       ]
            for output in execute(command):
                print(output)

            for atlas in ATLAS_TYPES:
                # Atlas 09c -> Subject space (using previous omat)
                command = ['flirt',
                           '-in',
                           opj(EXTERNAL,
                               'bha_' + atlas + '_1mm_mni09c.nii.gz'),
                           '-ref',
                           opj(PROCESSED, sub, ses, 'anat',
                               sub + '_' + ses + '_T1w_brain.nii.gz'),
                           '-out',
                           opj(PROCESSED, sub, ses, 'anat',
                               sub + '_' + ses + '_' + atlas + '.nii.gz'),
                           '-init',
                           opj(PROCESSED, sub, ses, 'anat',
                               '09c_2_' + sub + '_' + ses + '.mat'),
                           '-applyxfm', '-interp', 'nearestneighbour',
                           ]
                for output in execute(command):
                    print(output)
                atlas_with_all_rois(sub, ses, atlas, opj(PROCESSED, sub, ses,
                                                         'anat', sub + '_' +
                                                         ses + '_' + atlas +
                                                         '.nii.gz'))

    return


def atlas_with_all_rois(sub, ses, atlas, new_atlas_path):
    """
    Function to correct atlas after resampling (looses some rois),
    this function recovers those lost rois
    """

    atlas_old = opj(EXTERNAL, 'bha_' + atlas + '_1mm_mni09c.nii.gz')
    atlas_new = opj(PROCESSED, sub, ses, 'func', sub + '_' + ses +
                    '_' + atlas + '_bold_space.nii.gz')

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
    nib.save(atlas_new_data_img_corrected,
             opj(PROCESSED, sub, ses, 'func', sub + '_' + ses +
                 '_' + atlas + '_bold_space.nii.gz'))


def extend_elec_location(elec_location):

    for elec_key, value in elec_location.items():
        x, y, z = value[0]
        elec_location[elec_key].append([x + 1, y + 1, z + 1])
        elec_location[elec_key].append([x + 1, y + 1, z - 1])
        elec_location[elec_key].append([x + 1, y + 1, z])
        elec_location[elec_key].append([x + 1, y - 1, z + 1])
        elec_location[elec_key].append([x + 1, y - 1, z - 1])
        elec_location[elec_key].append([x + 1, y - 1, z])
        elec_location[elec_key].append([x + 1, y, z + 1])
        elec_location[elec_key].append([x + 1, y, z - 1])
        elec_location[elec_key].append([x + 1, y, z])

        elec_location[elec_key].append([x - 1, y + 1, z + 1])
        elec_location[elec_key].append([x - 1, y + 1, z - 1])
        elec_location[elec_key].append([x - 1, y + 1, z])
        elec_location[elec_key].append([x - 1, y - 1, z + 1])
        elec_location[elec_key].append([x - 1, y - 1, z - 1])
        elec_location[elec_key].append([x - 1, y - 1, z])
        elec_location[elec_key].append([x - 1, y, z + 1])
        elec_location[elec_key].append([x - 1, y, z - 1])
        elec_location[elec_key].append([x - 1, y, z])

        elec_location[elec_key].append([x, y + 1, z + 1])
        elec_location[elec_key].append([x, y + 1, z - 1])
        elec_location[elec_key].append([x, y + 1, z])
        elec_location[elec_key].append([x, y - 1, z + 1])
        elec_location[elec_key].append([x, y - 1, z - 1])
        elec_location[elec_key].append([x, y - 1, z])
        elec_location[elec_key].append([x, y, z + 1])
        elec_location[elec_key].append([x, y, z - 1])
    return elec_location


def writeDict(dict, filename, sep=','):
    with open(filename, "a") as f:
        for i in dict.keys():
            f.write('elec[\'' + i + '\'] = [' +
                    sep.join([str(x) for x in dict[i]]) + ']\n')


def t1w_electrodes_to_09c(subject_list):

    ses = 'electrodes'

    for sub in subject_list:
        """
        Extract brain from electrodes T1W -> this to BIDS
        """
        command = ['bet',
                   opj(DATA, 'raw', 'bids', sub, ses, 'electrodes.nii.gz'),
                   opj(DATA, 'raw', 'bids', sub, ses,
                       'electrodes_brain.nii.gz'),
                   '-B', '-f', '0.1', '-s', '-m',
                   ]

        for output in execute(command):
            print(output)

        """
        Atlas to subject space
        """
        command = ['flirt',
                   '-in',
                   opj(DATA, 'raw', 'bids', sub, ses,
                       'electrodes_brain.nii.gz'),
                   '-ref',
                   opj(EXTERNAL_MNI_09c,
                       'mni_icbm152_t1_tal_nlin_asym_09c_brain.nii'),
                   '-cost', 'mutualinfo',
                   '-out',
                   opj(DATA, 'raw', 'bids', sub, ses,
                       'electrodes_brain_09c.nii.gz')
                   ]

        for output in execute(command):
            print(output)


def load_elec_file(elec_file):

    elec = {}
    with open(elec_file) as f:
        content = f.readlines()

    for line in content:
        exec(line)

    return elec


def locate_electrodes(subject_list):
    """
    function to locate each electrode contact in a ROI.
    (might be the contact not to fall down in no ROI)
    """
    from collections import defaultdict

    ses = 'electrodes'

    for sub in subject_list:
        elec_file = opj(DATA, 'raw', 'bids', sub, ses, 'elec.loc')
        elec_location_mni09 = load_elec_file(elec_file)

        atlas_neig_comb = [[atlas, neighbours] for atlas in ATLAS_TYPES
                           for neighbours in NEIGHBOURS]

        for atlas, neighbours in atlas_neig_comb:

            atlas_file = opj(EXTERNAL, 'bha_' + atlas + '_1mm_mni09c.nii.gz')

            if neighbours:
                elec_location_mni09 = extend_elec_location(elec_location_mni09)

            atlas_data = nib.load(atlas_file).get_data()
            roi_number = np.unique(atlas_data).shape[0]-1
            roi_location_mni09 = defaultdict(set)

            for elec in elec_location_mni09.keys():
                for location in elec_location_mni09[elec]:
                    x, y, z = location
                    roi_location_mni09[elec].add(atlas_data[x, y, z].astype('int'))

            writeDict(roi_location_mni09,
                      opj(DATA, 'raw', 'bids', sub, ses,
                          sub + '_elec_' + str(roi_number) + '_rois_' +
                          str(neighbours) + '_neighbours.roi'))


def contacts_from_electrode(first_contact_pos, last_contact_pos, contact_num,
                            elec_name):

    x1, y1, z1 = first_contact_pos
    x2, y2, z2 = last_contact_pos

    difference = np.array([x2-x1, y2-y1, z2-z1])

    portion = difference / (contact_num - 1)

    for num in range(contact_num):
        point = portion * num
        print('elec[\'' + elec_name + '\'] = [' +
              np.array2string(np.round(first_contact_pos + point).astype(int),
                              separator=', ') +
              ']')

# elec_name = 'OIM1'
# contact_num = 12
# first_contact_pos = [78, 73, 78]
# last_contact_pos = [80, 26, 84]


def create_centroids(atlas):

    atlas_file = opj(EXTERNAL, 'bha_' + atlas + '_1mm_mni09c.nii.gz')
    atlas_data = nib.load(atlas_file).get_data()

    roi_number = np.unique(atlas_data).shape[0]

    centroids = np.zeros((roi_number, 3))

    for roi_num in range(roi_number):
        centroids[roi_num] = np.mean(np.argwhere(atlas_data == roi_num),
                                     axis=0)

    np.save(opj(EXTERNAL,
                'bha_' + atlas + '_1mm_mni09c_roi_centroids'),
            centroids)


def find_closest_roi(location, atlas):

    location = np.array(location)

    if not op.exists(opj(EXTERNAL,
                         'bha_' + atlas + '_1mm_mni09c_roi_centroids.npy')):
        create_centroids(atlas)
    else:
        centroids = np.load(opj(EXTERNAL,
                                'bha_' + atlas +
                                '_1mm_mni09c_roi_centroids.npy'))

    closest_roi = 0
    min_dist = np.inf
    for idx, centroid in enumerate(centroids):
        if idx:  # jump ROI == 0 (ROI == 0 is nothing)
            dist = np.linalg.norm(location - centroid)
            if min_dist > dist:
                min_dist = dist
                closest_roi = idx

    return closest_roi


def locate_electrodes_closest_roi(subject_list):
    """
    function to locate each electrode contact to the closest ROI.
    (might be the contact not to fall down in no ROI)
    """
    from collections import defaultdict

    ses = 'electrodes'

    for sub in subject_list:
        elec_file = opj(DATA, 'raw', 'bids', sub, ses, 'elec.loc')
        elec_location_mni09 = load_elec_file(elec_file)

        for atlas in ATLAS_TYPES:

            atlas_file = opj(EXTERNAL, 'bha_' + atlas + '_1mm_mni09c.nii.gz')

            atlas_data = nib.load(atlas_file).get_data()
            roi_number = np.unique(atlas_data).shape[0]-1
            roi_location_mni09 = defaultdict(set)

            for elec in elec_location_mni09.keys():
                for location in elec_location_mni09[elec]:
                    x, y, z = location
                    roi = atlas_data[x, y, z].astype('int')
                    if roi:
                        roi_location_mni09[elec].add(roi)
                    else:
                        closest_roi = find_closest_roi(location, atlas)
                        roi_location_mni09[elec].add(closest_roi)

            writeDict(roi_location_mni09,
                      opj(DATA, 'raw', 'bids', sub, ses,
                          sub + '_elec_' + str(roi_number) + '_closest_rois' +
                          '.roi'))










"""
from nilearn import datasets

dataset = datasets.fetch_atlas_harvard_oxford('cort-maxprob-thr25-2mm')
atlas_filename = dataset.maps
labels = dataset.labels

print('Atlas ROIs are located in nifti image (4D) at: %s' %
      atlas_filename)  # 4D data

# One subject of resting-state data
data = datasets.fetch_adhd(n_subjects=1)
fmri_filenames = data.func[0]


from nilearn.input_data import NiftiLabelsMasker
masker = NiftiLabelsMasker(labels_img=atlas_filename, standardize=True,
                           memory='nilearn_cache', verbose=5)

# Here we go from nifti files to the signal time series in a numpy
# array. Note how we give confounds to be regressed out during signal
# extraction
time_series = masker.fit_transform(fmri_filenames, confounds=data.confounds)

from nilearn.connectome import ConnectivityMeasure
correlation_measure = ConnectivityMeasure(kind='correlation')
correlation_matrix = correlation_measure.fit_transform([time_series_2754])[0]


# Plot the correlation matrix
import numpy as np
from matplotlib import pyplot as plt
plt.figure(figsize=(10, 10))
# Mask the main diagonal for visualization:
np.fill_diagonal(correlation_matrix, 0)

plt.imshow(correlation_matrix, interpolation="nearest", cmap="RdBu_r",
           vmax=0.8, vmin=-0.8)

# Add labels and adjust margins
plt.gca().yaxis.tick_right()
plt.subplots_adjust(left=.01, bottom=.3, top=.99, right=.62)
"""

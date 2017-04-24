# -*- coding: utf-8 -*-

"""
In Slicer:
"""
slicer.util.loadVolume("/home/asier/Desktop/ruber_data/colocacion electrodos/3099986/5331/00010001")
node = slicer.util.getNode("00010001_1")
filename = '/home/asier/Desktop/ruber_data/colocacion electrodos/3099986/SaveTests.nii.gz'
slicer.util.saveNode(node, filename)

"""
Spyder Editor

This is a temporary script file.
"""

from nilearn.image import resample_img
import nibabel as nib
import numpy as np


img = nib.load('/home/asier/Desktop/ruber_data/colocacion electrodos/3099986/SaveTests.nii.gz')
img.affine

img_data = img.get_data()
img_data.shape

corrected_img = np.squeeze(img_data[:,:,:,0,0])

img = nib.Nifti1Image(corrected_img, img.affine)

resampled_img = resample_img(img, target_affine=np.diag((1, 1, 1)))

nib.save(resampled_img, '/home/asier/Desktop/ruber_data/colocacion electrodos/test.nii.gz')

"""
Anonimize
"""

# cd Desktop/ruber_data/colocacion\ electrodos
#gdcmanon --dumb --empty 10,10 --empty 10,20 -i 3099986 -o 3099986_anon -r -V --continue
#
#gdcmanon --dumb --empty 10,10 --empty 10,20 --remove 10,40 --remove 10,1010 --replace 10,1030,10 -i 3099986 -o 3099986_anon -r -V --continue


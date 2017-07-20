tracts_file = '/home/asier/Desktop/test_track/tracts.Bfloat_2514';
nifti_reference_file = '/home/asier/Desktop/test_track/eddy_corrected_avg_b0.nii.gz';
atlas_file = '/home/asier/Desktop/test_track/rsub-001_ses-presurg_atlas_2514.nii.gz';
out_folder = '/home/asier/Desktop/test_track/out/';
num_nodes = 2514;

[fibers]=camino_tracts_reader(tracts_file,nifti_reference_file);

fibers_to_scnetworks_startend(fibers,atlas_file,num_nodes,out_folder)

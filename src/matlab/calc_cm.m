function calc_cm(tracts_file, reference_file, atlas_file, num_nodes, out_folder)


[fibers]=camino_tracts_reader(tracts_file, reference_file);
fibers_to_scnetworks_startend(fibers,atlas_file,num_nodes,out_folder)

end

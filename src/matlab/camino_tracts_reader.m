function [fibers]=camino_tracts_reader(tracts_file,nifti_reference_file)

%Read nifti file header and store the pixel sizes and transformation matrix
hdr=read_hdr(nifti_reference_file);
pixsize=hdr.scales;
T=[hdr.hist.srow; 0 0 0 1];
clear hdr;

%Read the track file

f = fopen(tracts_file,'r', 'b');
fend = 0;
i = 1; %index
while (fend == 0)
	%read streamline from input file
	a = fread(f,1,'float');
	if (isempty(a))
	  fend = 1;
 	  break;
	end
	b = fread(f,1,'float');
        Npoints = floor(a);

	index  = round(b);
	xyz = fread(f,[3 Npoints], 'float');	
	
	fibers{i} = mni2cor(xyz',T);
	i = i+1;
end
fclose(f);

disp (['Total number of streamlines: ' num2str(i)]);

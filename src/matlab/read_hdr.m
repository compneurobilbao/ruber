function [hdr]=read_hdr(nii_file)
% Read a NIfTI header.
%
% USAGE: [hdr]=read_hdr(nii_file)
% INPUTS:
%	nii_file: path of the NIfTI header data to be read.

[pathstr,name,ext] = fileparts(nii_file);

if(strcmp(ext,'.gz'))
    [pathstr,name2,ext] = fileparts([pathstr filesep name]);
    if(strcmp(ext,'.img') | strcmp(ext,'.nii') | strcmp(ext,'.hdr'))
        name = name2;
    end
elseif(~strcmp(ext,'.nii')&~strcmp(ext,'.hdr')&~strcmp(ext,'.img'))
    %NAME1 = [PATHSTR filesep NAME1 EXT];
else
  error('Unknow file format');
end

tmpname = tempname;
command = sprintf('sh -c ". ${FSLDIR}/etc/fslconf/fsl.sh; FSLOUTPUTTYPE=NIFTI_PAIR; export FSLOUTPUTTYPE; $FSLDIR/bin/fslmaths %s %s"\n', nii_file, tmpname);
system(command);



%%%%%%%%%%%%%%%%%%%%%%  Read header (hdr) %%%%%%%%%%%%%%%%%%%%%%%%
fnhdr=strcat(tmpname,'.hdr');

% open file in big-endian
endian='b';
fid=fopen(fnhdr,'r','b');
testval = fread(fid,1,'int32');
% check if this gives the correct header size - if not use little-endian
if (testval~=348),
  fclose(fid);
  fid=fopen(fnhdr,'r','l');
  endian='l';
  testval = fread(fid,1,'int32');
  if (testval~=348),
    error('Can not read this file format');
    return;
  end
end

dummy=fread(fid,36,'char');
num_dimensions=fread(fid,1,'int16');
if num_dimensions>7
   num_dimensions=7;
end  
dims=fread(fid,num_dimensions,'int16'); 
dummy=fread(fid,7-num_dimensions,'int16'); 
dummy=fread(fid,14,'char');
datatype=fread(fid,1,'int16'); 
bpp=fread(fid,1,'int16'); 
slice_start=fread(fid,1,'int16'); 
scale0=fread(fid,1,'float');
scales=fread(fid,num_dimensions,'float');
dummy=fread(fid,7-num_dimensions,'float');
vox_offset=fread(fid,1,'float');
scl_slope=fread(fid,1,'float');
scl_inter=fread(fid,1,'float');
slice_end=fread(fid,1,'short');
slice_code=fread(fid,1,'char');
xyzt_units=fread(fid,1,'char');
cal_max=fread(fid,1,'float');
cal_min=fread(fid,1,'float');
slice_duration=fread(fid,1,'float');
toffset=fread(fid,1,'float');
glmax=fread(fid,1,'int32');
glmin=fread(fid,1,'int32');
descrip=fread(fid,80,'char');
aux_file=fread(fid,24,'char');
qform_code=fread(fid,1,'short');
sform_code=fread(fid,1,'short');
quatern_b=fread(fid,1,'float');
quatern_c=fread(fid,1,'float');
quatern_d=fread(fid,1,'float');
qoffset_x=fread(fid,1,'float');
qoffset_y=fread(fid,1,'float');
qoffset_z=fread(fid,1,'float');
srow_x=fread(fid,4,'float');
srow_y=fread(fid,4,'float');
srow_z=fread(fid,4,'float');
intent_name=fread(fid,16,'char');
magic_=fread(fid,4,'char');

fclose(fid);
history=struct('descrip', deblank(char(descrip')), 'aux_file', deblank(char(aux_file')), 'qform_code', qform_code, 'sform_code', sform_code, 'quatern', [quatern_b quatern_c quatern_d], 'qoffset', [qoffset_x qoffset_y qoffset_z], 'srow', [srow_x  srow_y srow_z]', 'intent_name', deblank(char(intent_name')), 'magic', deblank(char(magic_')) );


hdr=struct('dims',dims,'scale0',scale0,'scales',scales, 'datatype', datatype,'bpp',bpp,'slice_start', slice_start,'vox_offset', vox_offset,'scl_slope',scl_slope,'scl_inter', scl_inter, 'slice_end', slice_end,'slice_code', slice_code, 'xyzt_units', xyzt_units,  'hist', history, 'endian',endian);


  
% cross platform compatible deleting of files
delete([tmpname,'.hdr']);
delete([tmpname,'.img']);

end

function fibers_to_scnetworks_startend(fibers,atlas_file,num_nodes,out_folder)

%Read atlas file
[hdr atlas]=read_nii(atlas_file);
XLim=hdr.dims(1); YLim=hdr.dims(2); ZLim=hdr.dims(3);
Voxel_size=hdr.scales';


%num_nodes=max(atlas(:));
fiber_num = zeros(num_nodes,num_nodes); %fiber number connecting 2 rois
num_voxels=zeros(num_nodes,num_nodes); %Number of voxels connecting 2 rois
mean_length=zeros(num_nodes,num_nodes); %Mean Length in mm of the fibers connecting 2 rois

fprintf('   Computing statistcs:              %d/%d',1,length(fibers));
for i = 1:length(fibers)
    for k=1:(1+length(num2str(length(fibers)))+length(num2str(i)))
      fprintf('\b');
    end
    fprintf('%d/%d',i,length(fibers));
    
    pStart = round(fibers{i}(1,:)+1);
    pEnd = round(fibers{i}(end,:)+1);
    
    if pStart(1)>0 && pStart(1)<=XLim &&  pStart(2)>0 && pStart(2)<=YLim && pStart(3)>0 && pStart(3)<=ZLim && pEnd(1)>0 && pEnd(1)<=XLim && pEnd(2)>0 && pEnd(2)<=YLim && pEnd(3)>0 && pEnd(3)<=ZLim
        roi_m = atlas(pStart(1),pStart(2),pStart(3));
        roi_n = atlas(pEnd(1),pEnd(2),pEnd(3));
        Index_Start=sub2ind(size(atlas),pStart(1),pStart(2),pStart(3));
        Index_End=sub2ind(size(atlas),pEnd(1),pEnd(2),pEnd(3));
        
        %for each fiber conecting 2 different rois
        if roi_m > 0 && roi_n > 0 && roi_m ~= roi_n && roi_m <= num_nodes && roi_n <= num_nodes
            %Number of fibers conecting 2 rois
            fiber_num(roi_m,roi_n) = fiber_num(roi_m,roi_n) + 1;
            fiber_num(roi_n,roi_m) = fiber_num(roi_n,roi_m) + 1;
            
 	    %for each voxel of the fiber
            for j=1:size(fibers{i},1) 
                 point(j,:)=round(fibers{i}(j,:)+1);
                 if point(j,1)>0 && point(j,1)<=XLim &&  point(j,2)>0 && point(j,2)<=YLim && point(j,3)>0 && point(j,3)<=ZLim

                     %Number of voxel conectings 2 rois
                     num_voxels(roi_m,roi_n)=num_voxels(roi_m,roi_n)+1;
                     num_voxels(roi_n,roi_m)=num_voxels(roi_n,roi_m)+1;
                
                 end
             end

             %Sum of ecuclidina distances in mm of each voxel in the fiber
            mean_length(roi_m,roi_n)=mean_length(roi_m,roi_n)+sum(sqrt(sum((fibers{i}(1:end-1,:).*repmat(Voxel_size,(length(fibers{i})-1),1)-fibers{i}(2:end,:).*repmat(Voxel_size,(length(fibers{i})-1),1)).^2,2)));
            mean_length(roi_n,roi_m)=mean_length(roi_m,roi_n);
        end     
    end
end
fprintf('\n');

for i=1:num_nodes
    for j=1:num_nodes
        if mean_length(i,j)~=0
               mean_length(i,j)=mean_length(i,j)/fiber_num(i,j); 
        end
    end

end

save([out_folder '/fiber_number.txt'],'fiber_num','-ascii');
save([out_folder '/num_voxels.txt'],'num_voxels','-ascii');
save([out_folder '/mean_length.txt'],'mean_length','-ascii');




clear fiber_num mean_FA mean_MD mean_AD mean_RD num_voxels mean_length;

euclidean_distance=zeros(num_nodes,num_nodes);
centroids=zeros(num_nodes,3);
for i=1:num_nodes
  aux=find(atlas==i);
  area_i=length(aux);
  [x,y,z]=ind2sub(size(atlas),aux);
  centroids(i,:)=sum([x y z])./area_i;
  centroids(i,:)=centroids(i,:).*Voxel_size;
end


for i=1:num_nodes
     euclidean_distance(i,:)=sqrt(sum((centroids-repmat(centroids(i,:),num_nodes,1)).^2,2));
end

save([out_folder '/euclidean_distance.txt'],'euclidean_distance','-ascii');




return

function assign_data_to_parcel_cifti_V2(data,parcel_cifti_file,outputdir,outputname)
% Assign data to watersheds, LR together, TOL 05/2014

ind_parcels_orig = ft_read_cifti_mod(parcel_cifti_file);
ind_parcels = ind_parcels_orig.data(1:59412,:);
ind_parcel_assgns = zeros(size(ind_parcels_orig.data));
parcel_num = unique(ind_parcels);
parcel_num(parcel_num == 0) = [];
for p = 1:length(parcel_num)
    ind_parcel_assgns(ind_parcels == parcel_num(p)) = data(p);
end

%indparcel_assgns(size(indparcel_assgns,1):66697,1) = 0;
fname = [outputdir outputname];
cifti_write_wHDR(ind_parcel_assgns, parcel_cifti_file, fname, 'dtseries')
%cifti_write_wHDR(ind_parcel_assgns, [], fname, 'dlabel')
%network_assignment_path = [outputdir outputname '.dtseries.nii']; 
%ft_write_cifti_mod(network_assignment_path, ind_parcel_assgns);
%attribute_gifti_data_to_cifti([outputdir '/' outputname '_L.func.gii'], [outputdir '/' outputname '_R.func.gii'], [outputdir '/' outputname '.dtseries.nii'])



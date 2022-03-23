clear all;
%% 
% make FC matrices per run per sub per session each file will hold 4 runs 
datalist = '/projects/b1081/member_directories/aporter/TaskRegScripts/SurfaceResiduals/batch_all.txt'; %holds all subs
FCDir = '/projects/b1081/iNetworks/Nifti/derivatives/preproc_FCProc_20.2.0/';
%for now dont think I need these
%load('better_jet_colormap.mat')
%atlas_dir = '/projects/b1081/Atlases/Evan_parcellation/';
%atlas_params = atlas_parameters_GrattonLab('Parcels333',atlas_dir);
%putting into 333 space 
brain = ft_read_cifti_mod('/projects/b1081/Atlases/Evan_parcellation/Published_parcels/Parcels_LR.dtseries.nii');

%initalize these using datalist file 
%probs save them out as session level holding whatever amount within that
%session? so 4 file per sub each holding 4 runs of task 
%% 
dataInfo = readtable(datalist); %read datalist on single sub 
% session level as another loop
numdats = size(dataInfo.sub, 1);
run_nums{i}=str2double(regexp(dataInfo.runs{i},',','split'))';
sess_nums{i}=str2double(regexp(dataInfo.sess{i}))'; %this might need to be after the first loop
%might need this to account for task item?
%for sub = 1:numdats run thru all subjects 
%initialize subject
%for ses = 1:length(sess_nums{i}) something like that start with session
%initalize session
parcel_corrmat=[]; %if going with session level put parcel corrmat here 
%first 
for i = 1:length(run_nums{i})
    file = ['/path/to/file']; %change or add to txt file?
    run= ft_read_cifti_mod(file);
    run_data = run.data(1:59412,:);
    tmaskFile = ['path/to/mask']; %change or add to txt file?
    tmask = table2array(readtable(tmaskFile));
    masked_data = run_data(:, logical(tmask));
    parcels_data = [];
    
    for j=1:333 %putting in 333 space?
      parcels_data(j,:) = nanmean(masked_data(find(brain.data == j),:));
    end
    matrix = corr(parcels_data'); 
    zt=atanh(matrix);
    parcel_corrmat=cat(3, parcel_corrmat, zt);
end
%location for storing matrices 
saveName=[strcat('/projects/b1081/iNetworks/path/to/file/sub/', sub, '_', sess, '_parcel_corrmat.mat')]; 
save(saveName, 'parcel_corrmat');
clear parcel_corrmat task t zt;
%end
%end for double loops 
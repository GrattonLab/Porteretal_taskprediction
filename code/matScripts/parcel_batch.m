addpath /Users/Alexis/Desktop/BarchSZ/cifti-matlab-master

addpath /Applications/gifti-master

addpath /Users/Alexis/Applications/general_plotting
addpath /Users/Alexis/Applications/read_write_cifti/utilities


parcel='/Users/Alexis/Desktop/MSC_Alexis/analysis/code/matScripts/Parcels_LR.dtseries.nii';


dataF=['/Users/Alexis/Desktop/MSC_Alexis/analysis/data/mvpa_data/rest/MSC06_parcel_corrmat.mat']; %'/' subs{j} '.csv'];
data=load(dataF);
corrmat=data.parcel_corrmat;
MSC06=mean(corrmat,3);
dir=['/Users/Alexis/Desktop/MSC_Alexis/analysis/output/FigsFinal/'];
assign_data_to_parcel_cifti_V2(MSC06,parcel,dir, 'MSC06')
%{
subs={'MSC01','MSC02','MSC03','MSC04','MSC05','MSC06','MSC07','MSC10'};
task={'mem','semantic','motor','glass'};
folds={'0','1','2','3','4','5','6','7','8','9'}
%for i=1:length(task);
    for j=1:length(folds);
        dataF=['/Users/Alexis/Desktop/MSC_Alexis/analysis/output/results/acc/ALL/foldMSC06/fold' folds{j} '.csv']; %'/' subs{j} '.csv'];
        data=load(dataF);
        dir=['/Users/Alexis/Desktop/MSC_Alexis/analysis/output/results/acc/ALL/foldMSC06/'];
        assign_data_to_parcel_cifti_V2(data,parcel,dir, folds{j})
        clear dataF
        clear data 
     end
%end
%}


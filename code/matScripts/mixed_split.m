function mixed_split(sub)
    %load mixed timeseries
    mixedFile=['/Users/Alexis/Desktop/MSC_Alexis/analysis/data/mvpa_data/mixed/' sub '_parcel_timecourse.mat'];
    mixedFC=load(mixedFile);
    %use for MSC03 and MSC10 
    mixed_tmaskFile=['/Users/Alexis/Box/Quest_Backup/MSC/TaskFC/FCProc_' sub '_mixed_pass2_FDfilt/condindices.mat'];
    %mixed_tmaskFile=['/Users/Alexis/Box/Quest_Backup/MSC/TaskFC/FCProc_' sub '_mixed_pass2/condindices.mat'];
    mixedTmask=load(mixed_tmaskFile);
    %glass
    nsamples = size(mixedFC.parcel_time, 2);
    parcel_corrmat=[];
    for day=1:nsamples
        task = mixedFC.parcel_time{day}(logical(mixedTmask.TIndFin(day).AllGlass),:);
        if isempty(task)==1
            continue;
        end 
        t=corr(task);
        zt=atanh(t);
        parcel_corrmat=cat(3, parcel_corrmat, zt);
    end 
    saveName=[strcat('/Users/Alexis/Desktop/MSC_Alexis/analysis/data/mvpa_data/glass/', sub, '_parcel_corrmat.mat')];
    save(saveName, 'parcel_corrmat');
    clear parcel_corrmat task t zt;
    %semantic
    parcel_corrmat=[];
    for day=1:nsamples
        task = mixedFC.parcel_time{day}(logical(mixedTmask.TIndFin(day).AllSemantic),:);
        if isempty(task)==1
            continue;
        end 
        t=corr(task);
        zt=atanh(t);
        parcel_corrmat=cat(3, parcel_corrmat, zt);
    end 
    saveName=[strcat('/Users/Alexis/Desktop/MSC_Alexis/analysis/data/mvpa_data/semantic/', sub, '_parcel_corrmat.mat')];
    save(saveName, 'parcel_corrmat');
    clear nsamples parcel_corrmat task t zt;
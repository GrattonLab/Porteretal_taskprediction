function rest_fourth(sub)
filePath='/Users/Alexis/Desktop/MSC_Alexis/analysis/data/mvpa_data/rest/corrmats_timesplit/thirds';
%load rest timeseries
    restFile=['/Users/Alexis/Desktop/MSC_Alexis/analysis/data/mvpa_data/rest/' sub '_parcel_timecourse.mat'];
    restFC=load(restFile);
    %loop through all days
    %rest
    nsamples = size(restFC.parcel_time, 2);
    parcel_corrmat=[];
    for day=1:nsamples
        rest = restFC.parcel_time{day}(logical(restFC.tmask_all{day}),:);
        if isempty(rest)==1
            continue;
        end 
        %split into 4 
        %first 
        i=rest(1:end/4,:)
        t=corr(i);
        zt=atanh(t);
        parcel_corrmat=cat(3, parcel_corrmat, zt);
        clear t zt 
        %second
        j=rest(end/4+1:end/4+end/4, :)
        t=corr(j);
        zt=atanh(t);
        parcel_corrmat=cat(3, parcel_corrmat, zt);
        clear t zt
        %third 
        k=rest(end/4+end/4+1:end/4+end/4+end/4, :)
        t=corr(k);
        zt=atanh(t);
        parcel_corrmat=cat(3, parcel_corrmat, zt);
        clear t zt 
        %fourth
        l=rest(end/4+end/4+end/4+1:end, :)
        t=corr(l);
        zt=atanh(t);
        parcel_corrmat=cat(3, parcel_corrmat, zt);
        clear t zt 
        
    end 
    saveName=[strcat('/Users/Alexis/Desktop/MSC_Alexis/analysis/data/mvpa_data/rest/corrmats_timesplit/fourths/', sub, '_parcel_corrmat.mat')];
    save(saveName, 'parcel_corrmat');
function GroupNet_rest_split(num,sub)
%load rest timeseries
    restFile=['/Users/Alexis/Desktop/MSC_Alexis/analysis/data/IndNet_rest/allsubs_rest_corrmats_bysess_orig_INDformat.mat'];
    restFC=load(restFile);
    %loop through all days
    %rest
    nsamples = size(restFC.rest_ts_bysess, 1);
    parcel_corrmat=[];
    for day=1:nsamples
        rest = restFC.rest_ts_bysess{num,day};
        if isempty(rest)==1
            continue;
        end 
        %split into 4 
        %first 
        i=rest(1:end/4,:);
        t=corr(i);
        zt=atanh(t);
        parcel_corrmat=cat(3, parcel_corrmat, zt);
        clear t zt 
        %second
        j=rest(end/4+1:end/4+end/4, :);
        t=corr(j);
        zt=atanh(t);
        parcel_corrmat=cat(3, parcel_corrmat, zt);
        clear t zt
        %third 
        k=rest(end/4+end/4+1:end/4+end/4+end/4, :);
        t=corr(k);
        zt=atanh(t);
        parcel_corrmat=cat(3, parcel_corrmat, zt);
        clear t zt 
        %fourth
        l=rest(end/4+end/4+end/4+1:end, :);
        t=corr(l);
        zt=atanh(t);
        parcel_corrmat=cat(3, parcel_corrmat, zt);
        clear t zt 
        
    end 
    saveName=[strcat('/Users/Alexis/Desktop/MSC_Alexis/analysis/data/IndNet_rest/', sub, '_parcel_corrmat.mat')];
    save(saveName, 'parcel_corrmat');
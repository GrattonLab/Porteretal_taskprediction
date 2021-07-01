function parcel_corrmat=reframe(sub,frame,task)
    %frame=int64(frame);
    memFile=['/Users/Alexis/Desktop/MSC_Alexis/analysis/data/mvpa_data/' task '/' sub '_parcel_timecourse.mat'];
    memFC=load(memFile);    
    nsamples = size(memFC.parcel_time, 2);
    parcel_corrmat=[];
    for day=1:nsamples
        task = memFC.parcel_time{day}(logical(memFC.tmask_all{day}),:);
        if isempty(task)==1
            continue;
        elseif round(size(task,1))<frame
            continue; 
        end 
        %this will find out what the size of the time series is 
        %randomly pick a number between 1 and the max size
        %then if the starting point is too close to the max size
        %it will go back to the beginning and take out the remaining frames
        %this is both a combination of chunking and random sample
        maxSize=round(size(task,1));
        startingPoint=randi(maxSize);
        if startingPoint+frame>maxSize
            t1=task(startingPoint:maxSize,:);
            tsize=maxSize-startingPoint;
            fRemaining=frame-tsize-1;
            t2=task(1:fRemaining,:);
            task_min=[t1; t2];
        else
            task_min=task(startingPoint:frame+startingPoint,:);
        end 
        %complete random sampling
        t=corr(task_min);
        zt=atanh(t);
        parcel_corrmat=cat(3, parcel_corrmat, zt);
     end 
end

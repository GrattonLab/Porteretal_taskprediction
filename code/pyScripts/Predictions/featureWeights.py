from sklearn.model_selection import cross_validate
import numpy as np
import os
import sys
import pandas as pd
#import other python scripts for further anlaysis
import reshape
from sklearn.linear_model import RidgeClassifier
from sklearn.model_selection import LeaveOneOut
#import results
import warnings
warnings.filterwarnings("ignore")
# Initialization of directory information:
thisDir = os.path.expanduser('~/Desktop/Porteretal_taskprediction/')
dataDir = thisDir + 'data/corrmats/'
splitDict=dict([('MSC01',10),('MSC02',10),('MSC03',8),('MSC04',10),('MSC05',10),('MSC06',9),('MSC07',9),('MSC10',10)])

subList=['MSC01','MSC02','MSC03','MSC04','MSC05','MSC06','MSC07','MSC10']
taskList=['semantic','motor','mem','glass']
Parcel_params = reshape.loadParcelParams('Gordon333')
roi_sort = np.squeeze(Parcel_params['roi_sort'])
outDir = thisDir + 'output/results/Ridge/'

def allSubs():
    for task in taskList:
        for train_sub in subList:
            classifyCV(train_sub,task)

def classifyCV(sub, task):
    """
    Classifying same subjects (CV) along the same task

    Parameters
    -------------

    Returns
    -------------
    dfCV : DataFrame
        Dataframe consisting of average accuracy across all subjects

    """
    clf=RidgeClassifier()

    taskFC=reshape.permROI(dataDir+task+'/'+sub+'_parcel_corrmat.mat')
    restFC=reshape.permROI(dataDir+'rest/'+sub+'_parcel_corrmat.mat')
    folds=taskFC.shape[0]
    x_train, y_train=reshape.concateFC(taskFC, restFC)
    output = cross_validate(clf, x_train, y_train, cv=folds, return_estimator =True)
    session=0
    i=len(output['estimator'])
    arr=np.empty([i,55278])
    for model in output['estimator']:
        arr[session]=model.coef_
        session=session+1
    fwAve=arr.mean(axis=0)
    indices=reshape.getIndices()
    indices['fw']=fwAve
    lower_triang=indices[['level_0','level_1','variable_0','variable_1','fw']]
    lower_triang.rename(columns={'level_0':'variable_0','level_1':'variable_1','variable_0':'level_0','variable_1':'level_1'},inplace=True)
    full_mat=pd.concat([indices,lower_triang])
    features=full_mat.pivot(index=['level_0','level_1'],columns=['variable_0','variable_1'],values='fw')
    features.sort_index(axis=0,level=1,inplace=True)
    features.sort_index(axis=1,level=1,inplace=True)
    absolute=features.abs()
    dense_mat=absolute.sum(axis=1)
    data={'acc':dense_mat,'roi':roi_sort}
    df=pd.DataFrame(data)
    df.sort_values(by='roi',inplace=True)
    array=df['acc'].to_numpy()
    array.tofile(outDir+'single_task/fw/'+sub+'.csv', sep = ',')



def modelAll(train_sub):
    """
    Preparing machine learning model with appropriate data

    Parameters
    -------------
    train_sub : str
            Subject name for training
    test_sub : str
            Subject name for testing

    Returns
    -------------
    total_score : float
            Average accuracy of all folds

    """
    session=splitDict[train_sub]
    split=np.empty((session, 55278))
    count=0
    clf=RidgeClassifier()
    df=pd.DataFrame()
    #train sub
    memFC=reshape.permROI(dataDir+'mem/'+train_sub+'_parcel_corrmat.mat')
    semFC=reshape.permROI(dataDir+'semantic/'+train_sub+'_parcel_corrmat.mat')
    glassFC=reshape.permROI(dataDir+'glass/'+train_sub+'_parcel_corrmat.mat')
    motFC=reshape.permROI(dataDir+'motor/'+train_sub+'_parcel_corrmat.mat')
    restFC=reshape.permROI(dataDir+'rest/corrmats_timesplit/fourths/'+train_sub+'_parcel_corrmat.mat') #keep tasks seperated in order to collect the right amount of days
    restFC=np.reshape(restFC,(10,4,55278)) #reshape to gather correct days
    loo = LeaveOneOut()
    fw=np.empty([memFC.shape[0],55278])
    for train_index, test_index in loo.split(split):
        memtrain=memFC[train_index]
        semtrain=semFC[train_index]
        mottrain=motFC[train_index]
        glatrain=glassFC[train_index]
        Xtrain_task=np.concatenate((memtrain,semtrain,mottrain,glatrain))
        Xtrain_rest=restFC[train_index,:,:]
        Xtrain_rest=np.reshape(Xtrain_rest,(-1,55278))
        ytrain_task = np.ones(Xtrain_task.shape[0], dtype = int)
        ytrain_rest=np.zeros(Xtrain_rest.shape[0], dtype=int)
        X_tr=np.concatenate((Xtrain_task, Xtrain_rest))
        y_tr = np.concatenate((ytrain_task,ytrain_rest))
        clf.fit(X_tr,y_tr)
        features = clf.coef_[0]
        fw[count]=features
        count=count+1
    fwAve=fw.mean(axis=0)
    indices=reshape.getIndices()
    indices['fw']=fwAve
    lower_triang=indices[['level_0','level_1','variable_0','variable_1','fw']]
    lower_triang.rename(columns={'level_0':'variable_0','level_1':'variable_1','variable_0':'level_0','variable_1':'level_1'},inplace=True)
    full_mat=pd.concat([indices,lower_triang])
    features=full_mat.pivot(index=['level_0','level_1'],columns=['variable_0','variable_1'],values='fw')
    features.sort_index(axis=0,level=1,inplace=True)
    features.sort_index(axis=1,level=1,inplace=True)
    absolute=features.abs()
    dense_mat=absolute.sum(axis=1)
    data={'acc':dense_mat,'roi':roi_sort}
    df=pd.DataFrame(data)
    df.sort_values(by='roi',inplace=True)
    array=df['acc'].to_numpy()
    array.tofile(outDir+'/ALL_Binary/fw/'+train_sub+'.csv', sep = ',')


def allTasks():
    for train_sub in subList:
        modelAll(train_sub)



def allFolds(train_sub):
    session=splitDict[train_sub]
    split=np.empty((session, 55278))
    count=0
    clf=RidgeClassifier()
    df=pd.DataFrame()
    #train sub
    memFC=reshape.permROI(dataDir+'mem/'+train_sub+'_parcel_corrmat.mat')
    semFC=reshape.permROI(dataDir+'semantic/'+train_sub+'_parcel_corrmat.mat')
    glassFC=reshape.permROI(dataDir+'glass/'+train_sub+'_parcel_corrmat.mat')
    motFC=reshape.permROI(dataDir+'motor/'+train_sub+'_parcel_corrmat.mat')
    restFC=reshape.permROI(dataDir+'rest/corrmats_timesplit/fourths/'+train_sub+'_parcel_corrmat.mat') #keep tasks seperated in order to collect the right amount of days
    restFC=np.reshape(restFC,(10,4,55278)) #reshape to gather correct days
    loo = LeaveOneOut()
    fw=np.empty([memFC.shape[0],55278])
    for train_index, test_index in loo.split(split):
        memtrain=memFC[train_index]
        semtrain=semFC[train_index]
        mottrain=motFC[train_index]
        glatrain=glassFC[train_index]
        Xtrain_task=np.concatenate((memtrain,semtrain,mottrain,glatrain))
        Xtrain_rest=restFC[train_index,:,:]
        Xtrain_rest=np.reshape(Xtrain_rest,(-1,55278))
        ytrain_task = np.ones(Xtrain_task.shape[0], dtype = int)
        ytrain_rest=np.zeros(Xtrain_rest.shape[0], dtype=int)
        X_tr=np.concatenate((Xtrain_task, Xtrain_rest))
        y_tr = np.concatenate((ytrain_task,ytrain_rest))
        clf.fit(X_tr,y_tr)
        features = clf.coef_[0]
        fw[count]=features
        count=count+1
    fwSize=fw.shape[0]
    for i in range(fwSize):
        fold=fw[i]
        indices=reshape.getIndices()
        indices['fw']=fold
        lower_triang=indices[['level_0','level_1','variable_0','variable_1','fw']]
        lower_triang.rename(columns={'level_0':'variable_0','level_1':'variable_1','variable_0':'level_0','variable_1':'level_1'},inplace=True)
        full_mat=pd.concat([indices,lower_triang])
        features=full_mat.pivot(index=['level_0','level_1'],columns=['variable_0','variable_1'],values='fw')
        features.sort_index(axis=0,level=1,inplace=True)
        features.sort_index(axis=1,level=1,inplace=True)
        absolute=features.abs()
        dense_mat=absolute.sum(axis=1)
        data={'acc':dense_mat,'roi':roi_sort}
        df=pd.DataFrame(data)
        df.sort_values(by='roi',inplace=True)
        array=df['acc'].to_numpy()
        array.tofile(outDir+'ALL_Binary/fw/'+str(i)+'.csv', sep = ',')






def groupApp():
    """
    Feature weights for groupwise approach single task
    Parameters
    -------------


    Returns
    -------------
    dfGroup : DataFrame
        Dataframe consisting of group average accuracy training with subs instead of session

    """
    for task in taskList:
        FW_model(task)


def FW_model(train_task):
    """
    Preparing machine learning model with appropriate data

    Parameters
    -------------
    train_task : str
            Task name for training
    test_task : str
            Task name for testing

    Returns
    -------------
    within_score : float
            Feature weights averaged across all folds

    """

    #nsess x fc x nsub
    ds_T=np.empty((8,55278,8))
    ds_R=np.empty((8,55278,8))
    count=0
    #get all subs for a given task
    for sub in subList:
        #training task
        tmp_taskFC=reshape.permROI(dataDir+train_task+'/'+sub+'_parcel_corrmat.mat')
        tmp_taskFC=tmp_taskFC[:8,:]
        tmp_restFC=reshape.permROI(dataDir+'rest/'+sub+'_parcel_corrmat.mat')
        tmp_restFC=tmp_restFC[:8,:]
        #reshape 2d into 3d nsessxfcxnsubs
        ds_T[:,:,count]=tmp_taskFC
        ds_R[:,:,count]=tmp_restFC
        count=count+1
    clf=RidgeClassifier()
    loo = LeaveOneOut()
    wtn_scoreList=[]
    all_subs_features=np.empty([8,55278]) # 8 subs
    all_count = 0
    for i in range(8):
    #takes one session of data (7 subs)
        taskFC=ds_T[i,:,:]
        taskFC=taskFC.T
        restFC=ds_R[i,:,:]
        restFC=restFC.T
        taskSize=taskFC.shape[0]
        restSize=restFC.shape[0]
        t = np.ones(taskSize, dtype = int)
        r=np.zeros(restSize, dtype=int)
        fw=np.empty([taskFC.shape[0],55278])
        count_folds = 0
        for train_index, test_index in loo.split(taskFC):
            Xtrain_rest, Xtest_rest=restFC[train_index], restFC[test_index]
            Xtrain_task, Xtest_task=taskFC[train_index], taskFC[test_index]
            ytrain_rest,ytest_rest=r[train_index], r[test_index]
            ytrain_task,ytest_task=t[train_index], t[test_index]
            X_tr=np.concatenate((Xtrain_task, Xtrain_rest))
            y_tr = np.concatenate((ytrain_task,ytrain_rest))
            clf.fit(X_tr,y_tr)
            features = clf.coef_[0]
            fw[count_folds]=features
            count_folds=count_folds+1
        fwAve = fw.mean(axis = 0) #average for all folds
        all_subs_features[all_count] = fwAve
        all_count = all_count +1
    all_feats = all_subs_features.mean(axis = 0)
    indices=reshape.getIndices()
    indices['fw']=all_feats
    lower_triang=indices[['level_0','level_1','variable_0','variable_1','fw']]
    lower_triang.rename(columns={'level_0':'variable_0','level_1':'variable_1','variable_0':'level_0','variable_1':'level_1'},inplace=True)
    full_mat=pd.concat([indices,lower_triang])
    features=full_mat.pivot(index=['level_0','level_1'],columns=['variable_0','variable_1'],values='fw')
    features.sort_index(axis=0,level=1,inplace=True)
    features.sort_index(axis=1,level=1,inplace=True)
    absolute=features.abs()
    dense_mat=absolute.sum(axis=1)
    data={'acc':dense_mat,'roi':roi_sort}
    df=pd.DataFrame(data)
    df.sort_values(by='roi',inplace=True)
    array=df['acc'].to_numpy()
    array.tofile(outDir+'single_task/fw/'+train_task+'groupwise_fw.csv', sep = ',')



def allTask_FW():
    """
    Leave one subject out, using all tasks but only using one session per model; 28 samples 4Task*7subs*1session

    Parameters
    -------------

    Returns
    -------------
    df : pandas Dataframe
            Accuracy, sensitivity, specificity scores for each kfold

    """
    clf=RidgeClassifier()
    loo = LeaveOneOut()
    final_df=pd.DataFrame()
    ds_mem=np.empty((8,55278,8))
    ds_sem=np.empty((8,55278,8))
    ds_glass=np.empty((8,55278,8))
    ds_motor=np.empty((8,55278,8))
    ds_R=np.empty((8,55278,8))
    count=0
    #get all subs for a given task
    for sub in subList:
        #training task
        tmp_memFC=reshape.matFiles(dataDir+'mem/'+sub+'_parcel_corrmat.mat')
        tmp_memFC=tmp_memFC[:8,:]
        tmp_semFC=reshape.matFiles(dataDir+'semantic/'+sub+'_parcel_corrmat.mat')
        tmp_semFC=tmp_semFC[:8,:]
        tmp_glassFC=reshape.matFiles(dataDir+'glass/'+sub+'_parcel_corrmat.mat')
        tmp_glassFC=tmp_glassFC[:8,:]
        tmp_motorFC=reshape.matFiles(dataDir+'motor/'+sub+'_parcel_corrmat.mat')
        tmp_motorFC=tmp_motorFC[:8,:]
        tmp_restFC=reshape.matFiles(dataDir+'rest/'+sub+'_parcel_corrmat.mat')
        #reshape 2d into 3d nsessxfcxnsubs
        tmp_restFC=tmp_restFC[:8,:]
        ds_mem[:,:,count]=tmp_memFC
        ds_sem[:,:,count]=tmp_semFC
        ds_glass[:,:,count]=tmp_glassFC
        ds_motor[:,:,count]=tmp_motorFC
        ds_R[:,:,count]=tmp_restFC
        count=count+1
    #split up by subs not sess
    sub_splits=np.empty((8,55278))
    accScore_allsess=[]
    all_subs_features=np.empty([8,55278]) # 8 subs
    all_count = 0
    #declare session to work in first 10 sessions
    for i in range(8):
        cv_scoreList=[]
        tmp_mem=ds_mem[i,:,:]
        tmp_mem=tmp_mem.T

        tmp_mot=ds_motor[i,:,:]
        tmp_mot=tmp_mot.T

        tmp_sem=ds_sem[i,:,:]
        tmp_sem=tmp_sem.T

        tmp_glass=ds_glass[i,:,:]
        tmp_glass=tmp_glass.T

        ds_rest=ds_R[i,:,:]
        ds_rest=ds_rest.T
        fw=np.empty([8,55278])
        count_folds = 0
        for train_index, test_index in loo.split(sub_splits):
            #train on all sessions 1-6 subs
            #test on all sessions of one sub
            train_mem, test_mem=tmp_mem[train_index,:],tmp_mem[test_index,:]
            train_mot, test_mot=tmp_mot[train_index,:],tmp_mot[test_index,:]
            train_sem, test_sem=tmp_sem[train_index,:],tmp_sem[test_index,:]
            train_glass, test_glass=ds_rest[train_index,:],ds_rest[test_index,:]
            train_rest, test_rest=tmp_glass[train_index,:],tmp_glass[test_index,:]
            #now reshape data
            Xtrain_task=np.concatenate((train_mem,train_mot,train_sem,train_glass))
            Xtest_task=np.concatenate((test_mem,test_mot,test_sem,test_glass))
            #training set
            taskSize=Xtrain_task.shape[0]
            restSize=train_rest.shape[0]
            t = np.ones(taskSize, dtype = int)
            r=np.zeros(restSize, dtype=int)
            x_train=np.concatenate((Xtrain_task,train_rest))
            y_train=np.concatenate((t,r))
            #testing set (left out sub CV)
            testSize=Xtest_task.shape[0]
            test_restSize=test_rest.shape[0]
            test_t = np.ones(testSize, dtype = int)
            test_r=np.zeros(test_restSize, dtype=int)
            clf.fit(x_train,y_train)
            features = clf.coef_[0]
            fw[count_folds]=features
            count_folds=count_folds+1
        fwAve = fw.mean(axis = 0) #average for all folds
        all_subs_features[all_count] = fwAve
        all_count = all_count +1
    all_feats = all_subs_features.mean(axis = 0)
    indices=reshape.getIndices()
    indices['fw']=all_feats
    lower_triang=indices[['level_0','level_1','variable_0','variable_1','fw']]
    lower_triang.rename(columns={'level_0':'variable_0','level_1':'variable_1','variable_0':'level_0','variable_1':'level_1'},inplace=True)
    full_mat=pd.concat([indices,lower_triang])
    features=full_mat.pivot(index=['level_0','level_1'],columns=['variable_0','variable_1'],values='fw')
    features.sort_index(axis=0,level=1,inplace=True)
    features.sort_index(axis=1,level=1,inplace=True)
    absolute=features.abs()
    dense_mat=absolute.sum(axis=1)
    data={'acc':dense_mat,'roi':roi_sort}
    df=pd.DataFrame(data)
    df.sort_values(by='roi',inplace=True)
    array=df['acc'].to_numpy()
    array.tofile(outDir+'ALL_Binary/fw/groupwise_fw.csv', sep = ',')

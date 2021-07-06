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

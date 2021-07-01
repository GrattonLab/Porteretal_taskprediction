#!/usr/bin/env python
# coding: utf-8

# In[ ]:

from sklearn.model_selection import train_test_split
from sklearn.model_selection import LeaveOneOut
from sklearn.linear_model import RidgeClassifier
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score
import itertools
import scipy.io
import random
import os
import sys
#import other python scripts for further anlaysis
# Initialization of directory information:
thisDir = os.path.expanduser('~/Desktop/MSC_Alexis/analysis/')
dataDir = thisDir + 'data/mvpa_data/'
outDir = thisDir + 'output/results/rdmNetwork/'
#dataDir = '/projects/p31240/'
#outDir = '/projects/p31240/rdmNetwork/'
# Subjects and tasks
taskList=['semantic','glass', 'motor','mem']
subList=['MSC01','MSC02','MSC03','MSC04','MSC05','MSC06','MSC07','MSC10']
#all possible combinations of subs and tasks
subsComb=(list(itertools.permutations(subList, 2)))
tasksComb=(list(itertools.permutations(taskList, 2)))
#DS combination
DSvars=list(itertools.product(list(subsComb),list(taskList)))
##SS combination
SSvars=list(itertools.product(list(subList),list(tasksComb)))
#BS combination
BSvars=list(itertools.product(list(subsComb),list(tasksComb)))
#CV combinations
CVvars=list(itertools.product(list(subList),list(taskList)))




def matFiles(df='path'):
    """
    Convert matlab files into upper triangle np.arrays
    Parameters
    -----------
    df : str
        Path to file
    Returns
    -----------
    ds : 2D upper triangle FC measures in (roi, days) format

    """
    #Consistent parameters to use for editing datasets
    nrois=333
    #Load FC file
    fileFC=scipy.io.loadmat(df)

    #Convert to numpy array
    fileFC=np.array(fileFC['parcel_corrmat'])
    #Replace nans and infs with zero
    fileFC=np.nan_to_num(fileFC)
    nsess=fileFC.shape[2]
    #Index upper triangle of matrix
    mask=np.triu_indices(nrois,1)
    ds=np.empty((nsess, int(nrois*(nrois-1)/2)))
    count=0
    #Loop through all 10 days to reshape correlations into linear form
    for sess in range(nsess):
        tmp=fileFC[:,:,sess]
        ds[count]=tmp[mask]
        count=count+1
    mask = (ds == 0).all(1)
    column_indices = np.where(mask)[0]
    df = ds[~mask,:]
    return df

def randFeats(df, idx):
    """
    Random feature selection based on random indexing

    Parameters
    ----------
    df : str
        path to file
    idx : int
        number to index from
    Returns
    ----------
    featDS : Array of task or rest with random features selected
    """
    data=matFiles(df)
    feat=idx.shape[0]
    nsess=data.shape[0]
    featDS=np.empty((nsess, feat))
    for sess in range(nsess):
        f=data[sess][idx]
        featDS[sess]=f
    return featDS

def concateFC(taskFC, restFC):
    """
    Concatenates task and rest FC arrays and creates labels
    Parameters
    -----------
    taskFC, restFC : array_like
        Numpy arrays of FC upper triangle for rest and task
    Returns
    -----------
    x, y : array_like
        Arrays containing task and restFC concatenated together and labels for each
    """
    x=np.concatenate((taskFC, restFC))
    taskSize=taskFC.shape[0]
    restSize=restFC.shape[0]
    t = np.ones(taskSize, dtype = int)
    r=np.zeros(restSize, dtype=int)
    y = np.concatenate((t,r))
    return x, y

def classifyDS(feat, number):
    acc_scores_per_task=[]
    tmp_df=pd.DataFrame(DSvars, columns=['sub','task'])
    dfDS=pd.DataFrame()
    dfDS[['train_sub','test_sub']]=pd.DataFrame(tmp_df['sub'].tolist())
    dfDS['task']=tmp_df['task']
    for index, row in dfDS.iterrows():
        score=model(feat, train_sub=row['train_sub'], test_sub=row['test_sub'], train_task=row['task'], test_task=row['task'])
        acc_scores_per_task.append(score)
    dfDS['acc']=acc_scores_per_task
    dfDS['features']=number
    return dfDS
    #dfDS.to_csv(outDir+'DS/'+str(number)+'.csv', index=False)
def classifySS(feat, number):
    acc_scores_per_task=[]
    tmp_df=pd.DataFrame(SSvars, columns=['sub','task'])
    dfSS=pd.DataFrame()
    dfSS[['train_task','test_task']]=pd.DataFrame(tmp_df['task'].tolist())
    dfSS['sub']=tmp_df['sub']
    for index, row in dfSS.iterrows():
        score=model(feat, train_sub=row['sub'], test_sub=row['sub'], train_task=row['train_task'], test_task=row['test_task'])
        acc_scores_per_task.append(score)
    dfSS['acc']=acc_scores_per_task
    dfSS['features']=number
    #save accuracy
    return dfSS
    #dfSS.to_csv(outDir+'SS/'+str(number)+'.csv', index=False)
def classifyBS(feat, number):
    acc_scores_per_task=[]
    tmp_df=pd.DataFrame(BSvars, columns=['sub','task'])
    dfBS=pd.DataFrame()
    dfBS[['train_task','test_task']]=pd.DataFrame(tmp_df['task'].tolist())
    dfBS[['train_sub', 'test_sub']]=pd.DataFrame(tmp_df['sub'].tolist())
    for index, row in dfBS.iterrows():
        score=model(feat,train_sub=row['train_sub'], test_sub=row['test_sub'], train_task=row['train_task'], test_task=row['test_task'])
        acc_scores_per_task.append(score)
    dfBS['acc']=acc_scores_per_task
    dfBS['features']=number
    #save accuracy
    return dfBS
    #dfBS.to_csv(outDir+'BS/'+str(number)+'.csv', index=False)

def classifyCV(feat, number):
    dfCV=pd.DataFrame(CVvars, columns=['sub','task'])
    clf=RidgeClassifier()
    acc_scores_per_task=[]
    for index, row in dfCV.iterrows():
        taskFC= randFeats(dataDir+row['task']+'/'+row['sub']+'_parcel_corrmat.mat',feat)
        restFC= randFeats(dataDir+'rest/'+row['sub']+'_parcel_corrmat.mat',feat)
        folds=taskFC.shape[0]
        x_train, y_train= concateFC(taskFC, restFC)
        CVscores=cross_val_score(clf, x_train, y_train, cv=folds)
        mu=CVscores.mean()
        acc_scores_per_task.append(mu)
    #acc per fold per sub
    #average acc per sub per tasks
    dfCV['acc']=acc_scores_per_task
    dfCV['features']=number
    return dfCV

def model(feat,train_sub, test_sub, train_task, test_task):
    clf=RidgeClassifier()
    taskFC= randFeats(dataDir+train_task+'/'+train_sub+'_parcel_corrmat.mat',feat)
    #if your subs are the same
    if train_sub==test_sub:
        tmp_restFC= randFeats(dataDir+'rest/corrmats_timesplit/half/'+train_sub+'_parcel_corrmat.mat',feat)
        restFC=tmp_restFC[:10]
        test_restFC=tmp_restFC[10:]
        #restFC, test_restFC=train_test_split(restFC, test_size=.5)
        test_taskFC= randFeats(dataDir+test_task+'/'+test_sub+'_parcel_corrmat.mat',feat)
        ACCscores=CV_folds(clf, taskFC, restFC, test_taskFC, test_restFC)
    else:
        restFC= randFeats(dataDir+'rest/'+train_sub+'_parcel_corrmat.mat',feat)
        test_taskFC= randFeats(dataDir+test_task+'/'+test_sub+'_parcel_corrmat.mat',feat)
        test_restFC= randFeats(dataDir+'rest/'+test_sub+'_parcel_corrmat.mat',feat)
        ACCscores=CV_folds(clf, taskFC, restFC, test_taskFC, test_restFC)
    return ACCscores
def CV_folds(clf, taskFC, restFC, test_taskFC, test_restFC):
    loo = LeaveOneOut()
    taskSize=taskFC.shape[0]
    restSize=restFC.shape[0]
    t = np.ones(taskSize, dtype = int)
    r=np.zeros(restSize, dtype=int)
    test_taskSize=test_taskFC.shape[0]
    test_restSize=test_restFC.shape[0]
    test_t = np.ones(test_taskSize, dtype = int)
    test_r=np.zeros(test_restSize, dtype=int)
    Xtest=np.concatenate((test_taskFC,test_restFC))
    ytest=np.concatenate((test_t,test_r))
    df=pd.DataFrame()
    acc_score=[]
    #fold each training set
    for train_index, test_index in loo.split(taskFC):
        Xtrain_rest=restFC[train_index]
        Xtrain_task=taskFC[train_index]
        ytrain_rest=r[train_index]
        ytrain_task=t[train_index]
        X_tr=np.concatenate((Xtrain_task, Xtrain_rest))
        y_tr = np.concatenate((ytrain_task,ytrain_rest))
        clf.fit(X_tr,y_tr)
        #fold each testing set
        score=clf.score(Xtest,ytest)
        acc_score.append(score)
    df['outer_fold']=acc_score
    total_score=df['outer_fold'].mean()

    return total_score

def runScript():
    DS_df=pd.DataFrame()
    SS_df=pd.DataFrame()
    BS_df=pd.DataFrame()
    CV_df=pd.DataFrame()
    #generate log sample
    #1000 points for log selection
    #loop through 125 times to generate 8*125=1000 samples per log point
    featureSize=np.logspace(1, 4.7, num=39,dtype=int)
    for i in range(100):
        for number in featureSize:
                #generate a new index
            idx=np.random.randint(55278, size=(number))
            DS=classifyDS(idx, number)
            DS_df=pd.concat([DS_df,DS])
            SS=classifySS(idx, number)
            SS_df=pd.concat([SS_df,SS])
            BS=classifyBS(idx, number)
            BS_df=pd.concat([BS_df,BS])
            CV=classifyCV(idx, number)
            CV_df=pd.concat([CV_df,CV])
        print('Finished with '+str(i)+' in iteration 100')
    DS_df.to_csv(outDir+'DS/acc.csv', index=False)
    SS_df.to_csv(outDir+'SS/acc.csv', index=False)
    BS_df.to_csv(outDir+'BS/acc.csv', index=False)
    CV_df.to_csv(outDir+'CV/acc.csv', index=False)

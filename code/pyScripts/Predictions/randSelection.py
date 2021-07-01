#!/usr/bin/env python
# coding: utf-8

# In[ ]:

from sklearn.model_selection import train_test_split
from sklearn.model_selection import LeaveOneOut
from sklearn.linear_model import RidgeClassifier
import numpy as np
import os
import sys
import pandas as pd
from sklearn.model_selection import cross_val_score
import itertools
#import other python scripts for further anlaysis
import reshape
# Initialization of directory information:
thisDir = os.path.expanduser('~/Desktop/MSC_Alexis/analysis/')
dataDir = thisDir + 'data/mvpa_data/'
outDir = thisDir + 'output/results/rdmNetwork/'
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
        taskFC=reshape.randFeats(dataDir+row['task']+'/'+row['sub']+'_parcel_corrmat.mat',feat)
        restFC=reshape.randFeats(dataDir+'rest/'+row['sub']+'_parcel_corrmat.mat',feat)
        folds=taskFC.shape[0]
        x_train, y_train=reshape.concateFC(taskFC, restFC)
        CVscores=cross_val_score(clf, x_train, y_train, cv=folds)
        mu=CVscores.mean()
        acc_scores_per_task.append(mu)
    #acc per fold per sub
    #average acc per sub per tasks
    dfCV['acc']=acc_scores_per_task
    dfCV['features']=number
    return dfCV
    #dfCV.to_csv(outDir+'CV/'+str(number)+'.csv')

def model(feat,train_sub, test_sub, train_task, test_task):
    clf=RidgeClassifier()
    taskFC=reshape.randFeats(dataDir+train_task+'/'+train_sub+'_parcel_corrmat.mat',feat)
    #if your subs are the same
    if train_sub==test_sub:
        restFC=reshape.randFeats(dataDir+'rest/corrmats_timesplit/half/'+train_sub+'_parcel_corrmat.mat',feat)
        restFC, test_restFC=train_test_split(restFC, test_size=.5)
        test_taskFC=reshape.randFeats(dataDir+test_task+'/'+test_sub+'_parcel_corrmat.mat',feat)
        ACCscores=CV_folds(clf, taskFC, restFC, test_taskFC, test_restFC)
    else:
        restFC=reshape.randFeats(dataDir+'rest/'+train_sub+'_parcel_corrmat.mat',feat)
        test_taskFC=reshape.randFeats(dataDir+test_task+'/'+test_sub+'_parcel_corrmat.mat',feat)
        test_restFC=reshape.randFeats(dataDir+'rest/'+test_sub+'_parcel_corrmat.mat',feat)
        ACCscores=CV_folds(clf, taskFC, restFC, test_taskFC, test_restFC)
    return ACCscores
def CV_folds(clf, taskFC, restFC, test_taskFC, test_restFC):
    loo = LeaveOneOut()
    taskSize=taskFC.shape[0]
    restSize=restFC.shape[0]
    t = np.ones(taskSize, dtype = int)
    r=np.zeros(restSize, dtype=int)
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
        tmpdf=pd.DataFrame()
        acc_scores_per_fold=[]
        #fold each testing set
        for t_index, te_index in loo.split(test_taskFC):
            Xtest_rest=test_restFC[te_index]
            Xtest_task=test_taskFC[te_index]
            X_te=np.concatenate((Xtest_task, Xtest_rest))
            y_te=np.array([1, 0])
            #test set
            clf.predict(X_te)
            #Get accuracy of model
            ACCscores=clf.score(X_te,y_te)
            acc_scores_per_fold.append(ACCscores)
        tmpdf['inner_fold']=acc_scores_per_fold
        score=tmpdf['inner_fold'].mean()
        acc_score.append(score)
    df['outer_fold']=acc_score
    total_score=df['outer_fold'].mean()

    return total_score

#!/usr/bin/env python
# coding: utf-8

# In[ ]:
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import LeaveOneOut
from sklearn.linear_model import RidgeClassifier
import numpy as np
import os
import sys
import pandas as pd
import itertools
import reshape
from statistics import mean
import scipy.io
#import results
import warnings
warnings.filterwarnings("ignore")
# Initialization of directory information:
thisDir = os.path.expanduser('~/Desktop/Porteretal_taskprediction/')
dataDir = thisDir + 'data/corrmats/'
outDir = thisDir + 'output/results/'
figsDir=thisDir + 'output/figures/'
IndNetDir=thisDir+'data/IndNet/'
# Subjects and tasks
taskList=['pres1','pres2','pres3']
#omitting MSC06 for classify All
#subList=['MSC01','MSC02','MSC03','MSC04','MSC05','MSC07','MSC10']
subList=['MSC01','MSC02','MSC03','MSC04','MSC05','MSC06','MSC07','MSC10']
#all possible combinations of subs and tasks
subsComb=(list(itertools.permutations(subList, 2)))
#tasksComb=(list(itertools.permutations(taskList, 2)))
tasksComb=(list(itertools.combinations(taskList, 2)))
DSvars=list(itertools.product(list(subsComb),list(taskList)))
splitDict=dict([('MSC01',10),('MSC02',10),('MSC03',8),('MSC04',10),('MSC05',10),('MSC06',9),('MSC07',9),('MSC10',10)])
taskDict=dict([('mem',0),('motor',1),('glass',2),('semantic',3)])
SSvars=list(itertools.product(list(subList),list(tasksComb))) #same sub diff task
BSvars=list(itertools.product(list(subsComb),list(tasksComb))) #diff sub diff task


def classifydiffMem(classifier='string'):
    """
    Classifying different subjects (DS) along the same task

    Parameters
    -------------
    classifier : string
        Provide classifier type for analysis, options SVM=LinearSVC(), Log=LogisticRegression(solver='lbfgs'), Ridge=RidgeClassifier()

    Returns
    -------------
    dfDS : DataFrame
        Dataframe consisting of average accuracy across all subjects

    """
    clf=RidgeClassifier()
    same_sub_per_task=[] #SS same sub
    diff_sub_per_task=[] #OS other sub
    tmp_BS=pd.DataFrame(BSvars, columns=['sub','task'])
    dfDS=pd.DataFrame()
    dfDS[['train_task','test_task']]=pd.DataFrame(tmp_BS['task'].tolist())
    dfDS[['train_sub', 'test_sub']]=pd.DataFrame(tmp_BS['sub'].tolist())

    for index, row in dfDS.iterrows():
        taskFC=reshape.matFiles(dataDir+'mem/'+row['train_task']+'/'+row['train_sub']+'_parcel_corrmat.mat')
        restFC=reshape.matFiles(dataDir+'mem/'+row['test_task']+'/'+row['train_sub']+'_parcel_corrmat.mat')
        test_taskFC=reshape.matFiles(dataDir+'mem/'+row['train_task']+'/'+row['test_sub']+'_parcel_corrmat.mat')
        test_restFC=reshape.matFiles(dataDir+'mem/'+row['test_task']+'/'+row['test_sub']+'_parcel_corrmat.mat')
        SSacc, OSacc=folds(clf, taskFC, restFC, test_taskFC, test_restFC)
        same_sub_per_task.append(SSacc)
        diff_sub_per_task.append(OSacc)
    dfDS['diff_sub']=diff_sub_per_task
    dfDS['same_sub']=same_sub_per_task
    dfDS.to_csv(outDir+classifier+'/single_task/sep_mem_pres_accCG.csv',index=False)
def folds(clf,taskFC, restFC, test_taskFC, test_restFC):
    """
    Cross validation to train and test using nested loops

    Parameters
    -----------
    clf : obj
        Machine learning algorithm
    analysis : str
        Analysis type
    taskFC, restFC, test_taskFC, test_restFC : array_like
        Input arrays, training and testing set of task and rest FC
    Returns
    -----------
    total_score : float
        Average accuracy across folds
    acc_score : list
        List of accuracy for each outer fold
    """

    loo = LeaveOneOut()
    taskSize=taskFC.shape[0]
    restSize=restFC.shape[0]
    t = np.ones(taskSize, dtype = int)
    r=np.zeros(restSize, dtype=int)
    y_test_task=np.ones(test_taskFC.shape[0])
    y_test_rest=np.zeros(test_restFC.shape[0])
    ytest=np.concatenate((y_test_task,y_test_rest))
    Xtest=np.concatenate((test_taskFC,test_restFC))
    #Test same sub
    SS_acc=[]
    #Test other subs
    OS_acc=[]

    #fold each training set
    for train_index, test_index in loo.split(taskFC):
        Xtrain_rest,Xval_rest=restFC[train_index],restFC[test_index]
        Xtrain_task,Xval_task=taskFC[train_index], taskFC[test_index]
        ytrain_rest,yval_rest=r[train_index], r[test_index]
        ytrain_task,yval_task=t[train_index], t[test_index]
        X_tr=np.concatenate((Xtrain_task, Xtrain_rest))
        y_tr = np.concatenate((ytrain_task,ytrain_rest))
        X_val=np.concatenate((Xval_task, Xval_rest))
        y_val = np.concatenate((yval_task,yval_rest))
        clf.fit(X_tr,y_tr)
        #Same subject
        SSscores=clf.score(X_val,y_val)
        SS_acc.append(SSscores)
        #Other subject
        OSscores=clf.score(Xtest,ytest)
        OS_acc.append(OSscores)
    OStotal_acc=mean(OS_acc)
    SStotal_acc=mean(SS_acc)
    return SStotal_acc, OStotal_acc

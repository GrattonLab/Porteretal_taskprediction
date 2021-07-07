#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#!/usr/bin/env python
# coding: utf-8

# In[2]:
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.model_selection import LeaveOneOut
from sklearn.linear_model import RidgeClassifier
import numpy as np
import os
import sys
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn import metrics
import itertools
#import other python scripts for further anlaysis
import reshape
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.model_selection import LeaveOneOut
from sklearn.linear_model import RidgeClassifier
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score
import itertools
import scipy.io
import random
from sklearn.model_selection import KFold
import os
import sys
import reshape
from statistics import mean
thisDir = os.path.expanduser('~/Desktop/Porteretal_taskprediction/')
dataDir = thisDir +'data/corrmats/'
outDir = thisDir + 'output/results/Ridge/'
# Subjects and tasks
taskList=['semantic','glass', 'motor','mem']
subList=['MSC01','MSC02','MSC03','MSC04','MSC05','MSC06','MSC07','MSC10']
netRoi=dict([('unassign',14808),('default', 10824),('visual',8736),('fp', 4620),('dan',5264),('van',3151),('salience', 494),('co', 4060),('sm', 2375),('sm-lat', 316),('auditory', 564),('pmn',45),('pon',21)])

splitDict=dict([('MSC01',10),('MSC02',10),('MSC03',8),('MSC04',10),('MSC05',10),('MSC06',9),('MSC07',9),('MSC10',10)])

def DSmodel(idx):
    """
    Preparing machine learning model with appropriate data

    Parameters
    -------------
    analysis : string
            The type of analysis to be conducted
    train_sub : str
            Subject name for training
    test_sub : str
            Subject name for testing
    train_task : str
            Task name for training
    test_task : str
            Task name for testing

    Returns
    -------------
    total_score : float
            Average accuracy of all folds

    """
    #clf=LinearSVC()
    #clf=LogisticRegression(solver = 'lbfgs')
    clf=RidgeClassifier()
    DS=pd.DataFrame()
    data=np.array(['MSC01','MSC02','MSC03','MSC04','MSC05','MSC06','MSC07','MSC10'],dtype='<U61')
    loo = LeaveOneOut()
    for t in taskList:
        for  test, train in loo.split(data): #train on one sub test on the rest
            tmp=pd.DataFrame()
            train_sub=data[train]
            test_sub=data[test]
        #train sub
            taskFC=reshape.subNets(dataDir+t+'/'+train_sub[0]+'_parcel_corrmat.mat',idx)
            restFC=reshape.subNets(dataDir+'rest/'+train_sub[0]+'_parcel_corrmat.mat',idx) #keep tasks seperated in order to collect the right amount of days
            test_taskFC,test_restFC=AllSubFiles_DS(test_sub,t,idx)
            same_sub, diff_sub=folds(clf, taskFC,restFC, test_taskFC,test_restFC)
            tmp['train']=train_sub
            tmp['task']=t
            tmp['same_sub']=same_sub
            tmp['diff_sub']=diff_sub
            DS=pd.concat([DS,tmp])
    return DS
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
    test_taskSize=test_taskFC.shape[0]
    test_restSize=test_restFC.shape[0]
    ttest = np.ones(test_taskSize, dtype = int)
    rtest=np.zeros(test_restSize, dtype=int)
    X_test=np.concatenate((test_taskFC, test_restFC))
    y_test = np.concatenate((ttest,rtest))
    df=pd.DataFrame()
    CVacc=[]
    DSacc=[]

    #fold each training set
    for train_index, test_index in loo.split(taskFC):
        Xtrain_rest,Xval_rest=restFC[train_index],restFC[test_index]
        Xtrain_task,Xval_task=taskFC[train_index],taskFC[test_index]
        ytrain_rest,yval_rest=r[train_index],r[test_index]
        ytrain_task,yval_task=t[train_index],t[test_index]
        X_tr=np.concatenate((Xtrain_task, Xtrain_rest))
        y_tr = np.concatenate((ytrain_task,ytrain_rest))
        Xval=np.concatenate((Xval_task,Xval_rest))
        yval=np.concatenate((yval_task,yval_rest))
        scaler = preprocessing.StandardScaler().fit(X_tr)
        scaler.transform(X_tr)
        clf.fit(X_tr,y_tr)
        scaler.transform(Xval)
        scaler.transform(X_test)
        same=clf.score(Xval,yval)
        diff=clf.score(X_test,y_test)

        CVacc.append(same)
        DSacc.append(diff)
    same_sub=mean(CVacc)
    diff_sub=mean(DSacc)

    return same_sub, diff_sub


def AllSubFiles_DS(test_sub,task,idx):
    """
    Return task and rest FC all subs
    Parameters
    -----------
    test_sub: Array of testing subs
    Returns
    ------------
    taskFC, restFC : Array of task and rest FC of all testing subs
    """
    a_taskFC=reshape.subNets(dataDir+task+'/'+test_sub[0]+'_parcel_corrmat.mat',idx)
    a_restFC=reshape.subNets(dataDir+'rest/'+test_sub[0]+'_parcel_corrmat.mat',idx)

    b_taskFC=reshape.subNets(dataDir+task+'/'+test_sub[1]+'_parcel_corrmat.mat',idx)
    b_restFC=reshape.subNets(dataDir+'rest/'+test_sub[1]+'_parcel_corrmat.mat',idx)

    c_taskFC=reshape.subNets(dataDir+task+'/'+test_sub[2]+'_parcel_corrmat.mat',idx)
    c_restFC=reshape.subNets(dataDir+'rest/'+test_sub[2]+'_parcel_corrmat.mat',idx)

    d_taskFC=reshape.subNets(dataDir+task+'/'+test_sub[3]+'_parcel_corrmat.mat',idx)
    d_restFC=reshape.subNets(dataDir+'rest/'+test_sub[3]+'_parcel_corrmat.mat',idx)

    e_taskFC=reshape.subNets(dataDir+task+'/'+test_sub[4]+'_parcel_corrmat.mat',idx)
    e_restFC=reshape.subNets(dataDir+'rest/'+test_sub[4]+'_parcel_corrmat.mat',idx)

    f_taskFC=reshape.subNets(dataDir+task+'/'+test_sub[5]+'_parcel_corrmat.mat',idx)
    f_restFC=reshape.subNets(dataDir+'rest/'+test_sub[5]+'_parcel_corrmat.mat',idx)

    g_taskFC=reshape.subNets(dataDir+task+'/'+test_sub[6]+'_parcel_corrmat.mat',idx)
    g_restFC=reshape.subNets(dataDir+'rest/'+test_sub[6]+'_parcel_corrmat.mat',idx)


    taskFC=np.concatenate((a_taskFC,b_taskFC,c_taskFC,d_taskFC,e_taskFC,f_taskFC,g_taskFC))
    restFC=np.concatenate((a_restFC,b_restFC,c_restFC,d_restFC,e_restFC,f_restFC,g_restFC))

    return taskFC, restFC

def AllSubFiles(test_sub,idx):
    """
    Return task and rest FC all subs
    Parameters
    -----------
    test_sub: Array of testing subs
    Returns
    ------------
    taskFC, restFC : Array of task and rest FC of all testing subs
    """
    a_memFC=reshape.subNets(dataDir+'mem/'+test_sub[0]+'_parcel_corrmat.mat',idx)
    a_semFC=reshape.subNets(dataDir+'semantic/'+test_sub[0]+'_parcel_corrmat.mat',idx)
    a_glassFC=reshape.subNets(dataDir+'glass/'+test_sub[0]+'_parcel_corrmat.mat',idx)
    a_motFC=reshape.subNets(dataDir+'motor/'+test_sub[0]+'_parcel_corrmat.mat',idx)
    a_restFC=reshape.subNets(dataDir+'rest/corrmats_timesplit/fourths/'+test_sub[0]+'_parcel_corrmat.mat',idx)

    b_memFC=reshape.subNets(dataDir+'mem/'+test_sub[1]+'_parcel_corrmat.mat',idx)
    b_semFC=reshape.subNets(dataDir+'semantic/'+test_sub[1]+'_parcel_corrmat.mat',idx)
    b_glassFC=reshape.subNets(dataDir+'glass/'+test_sub[1]+'_parcel_corrmat.mat',idx)
    b_motFC=reshape.subNets(dataDir+'motor/'+test_sub[1]+'_parcel_corrmat.mat',idx)
    b_restFC=reshape.subNets(dataDir+'rest/corrmats_timesplit/fourths/'+test_sub[1]+'_parcel_corrmat.mat',idx)

    c_memFC=reshape.subNets(dataDir+'mem/'+test_sub[2]+'_parcel_corrmat.mat',idx)
    c_semFC=reshape.subNets(dataDir+'semantic/'+test_sub[2]+'_parcel_corrmat.mat',idx)
    c_glassFC=reshape.subNets(dataDir+'glass/'+test_sub[2]+'_parcel_corrmat.mat',idx)
    c_motFC=reshape.subNets(dataDir+'motor/'+test_sub[2]+'_parcel_corrmat.mat',idx)
    c_restFC=reshape.subNets(dataDir+'rest/corrmats_timesplit/fourths/'+test_sub[2]+'_parcel_corrmat.mat',idx)

    d_memFC=reshape.subNets(dataDir+'mem/'+test_sub[3]+'_parcel_corrmat.mat',idx)
    d_semFC=reshape.subNets(dataDir+'semantic/'+test_sub[3]+'_parcel_corrmat.mat',idx)
    d_glassFC=reshape.subNets(dataDir+'glass/'+test_sub[3]+'_parcel_corrmat.mat',idx)
    d_motFC=reshape.subNets(dataDir+'motor/'+test_sub[3]+'_parcel_corrmat.mat',idx)
    d_restFC=reshape.subNets(dataDir+'rest/corrmats_timesplit/fourths/'+test_sub[3]+'_parcel_corrmat.mat',idx)

    e_memFC=reshape.subNets(dataDir+'mem/'+test_sub[4]+'_parcel_corrmat.mat',idx)
    e_semFC=reshape.subNets(dataDir+'semantic/'+test_sub[4]+'_parcel_corrmat.mat',idx)
    e_glassFC=reshape.subNets(dataDir+'glass/'+test_sub[4]+'_parcel_corrmat.mat',idx)
    e_motFC=reshape.subNets(dataDir+'motor/'+test_sub[4]+'_parcel_corrmat.mat',idx)
    e_restFC=reshape.subNets(dataDir+'rest/corrmats_timesplit/fourths/'+test_sub[4]+'_parcel_corrmat.mat',idx)

    f_memFC=reshape.subNets(dataDir+'mem/'+test_sub[5]+'_parcel_corrmat.mat',idx)
    f_semFC=reshape.subNets(dataDir+'semantic/'+test_sub[5]+'_parcel_corrmat.mat',idx)
    f_glassFC=reshape.subNets(dataDir+'glass/'+test_sub[5]+'_parcel_corrmat.mat',idx)
    f_motFC=reshape.subNets(dataDir+'motor/'+test_sub[5]+'_parcel_corrmat.mat',idx)
    f_restFC=reshape.subNets(dataDir+'rest/corrmats_timesplit/fourths/'+test_sub[5]+'_parcel_corrmat.mat',idx)

    g_memFC=reshape.subNets(dataDir+'mem/'+test_sub[6]+'_parcel_corrmat.mat',idx)
    g_semFC=reshape.subNets(dataDir+'semantic/'+test_sub[6]+'_parcel_corrmat.mat',idx)
    g_glassFC=reshape.subNets(dataDir+'glass/'+test_sub[6]+'_parcel_corrmat.mat',idx)
    g_motFC=reshape.subNets(dataDir+'motor/'+test_sub[6]+'_parcel_corrmat.mat',idx)
    g_restFC=reshape.subNets(dataDir+'rest/corrmats_timesplit/fourths/'+test_sub[6]+'_parcel_corrmat.mat',idx)


    taskFC=np.concatenate((a_memFC,a_semFC,a_glassFC,a_motFC,b_memFC,b_semFC,b_glassFC,b_motFC,c_memFC,c_semFC,c_glassFC,c_motFC,d_memFC,d_semFC,d_glassFC,d_motFC,e_memFC,e_semFC,e_glassFC,e_motFC,f_memFC,f_semFC,f_glassFC,f_motFC,g_memFC,g_semFC,g_glassFC,g_motFC))
    restFC=np.concatenate((a_restFC,b_restFC,c_restFC,d_restFC,e_restFC,f_restFC,g_restFC))

    return taskFC, restFC


def modelAll(idx):
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
    #clf=LinearSVC()
    #clf=LogisticRegression(solver = 'lbfgs')
    clf=RidgeClassifier()
    #train sub
    master_df=pd.DataFrame()
    data=np.array(['MSC01','MSC02','MSC03','MSC04','MSC05','MSC06','MSC07','MSC10'],dtype='<U61')
    loo = LeaveOneOut()
    for  test, train in loo.split(data): #train on one sub test on the rest
        tmp=pd.DataFrame()
        train_sub=data[train]
        test_sub=data[test]
    #train sub
        memFC=reshape.subNets(dataDir+'mem/'+train_sub[0]+'_parcel_corrmat.mat',idx)
        semFC=reshape.subNets(dataDir+'semantic/'+train_sub[0]+'_parcel_corrmat.mat',idx)
        glassFC=reshape.subNets(dataDir+'glass/'+train_sub[0]+'_parcel_corrmat.mat',idx)
        motFC=reshape.subNets(dataDir+'motor/'+train_sub[0]+'_parcel_corrmat.mat',idx)
        restFC=reshape.subNets(dataDir+'rest/corrmats_timesplit/fourths/'+train_sub[0]+'_parcel_corrmat.mat',idx) #keep tasks seperated in order to collect the right amount of days
        nsize=restFC.shape[1]
        restFC=np.reshape(restFC,(10,4,nsize)) #reshape to gather correct days
        #test sub
        test_taskFC,test_restFC=AllSubFiles(test_sub,idx)
        same_sub, diff_sub=foldsBinary(train_sub, clf, memFC,semFC,glassFC,motFC, restFC, test_taskFC,test_restFC)
        tmp['train']=train_sub
        tmp['same_sub']=same_sub
        tmp['diff_sub']=diff_sub
        master_df=pd.concat([master_df,tmp])
    return master_df

def foldsBinary(train_sub, clf, memFC,semFC,glassFC,motFC,restFC, test_taskFC, test_restFC):
    """
    Cross validation to train and test using nested loops

    Parameters
    -----------
    clf : obj
        Machine learning algorithm
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
    number=memFC.shape[1]
    test_taskSize=test_taskFC.shape[0]
    test_restSize=test_restFC.shape[0]
    testT= np.ones(test_taskSize, dtype = int)
    testR= np.zeros(test_restSize, dtype = int)
    Xtest=np.concatenate((test_taskFC,test_restFC))
    ytest=np.concatenate((testT,testR))
    CVacc=[]
    session=splitDict[train_sub[0]]
    split=np.empty((session, 55278))
    df=pd.DataFrame()
    DSacc=[]
    #fold each training set
    for train_index, test_index in loo.split(split):
        memtrain, memval=memFC[train_index], memFC[test_index]
        semtrain, semval=semFC[train_index], semFC[test_index]
        mottrain, motval=motFC[train_index], motFC[test_index]
        glatrain, glaval=glassFC[train_index], glassFC[test_index]
        Xtrain_task=np.concatenate((memtrain,semtrain,mottrain,glatrain))
        Xtrain_rest, Xval_rest=restFC[train_index,:,:], restFC[test_index,:,:]
        Xval_task=np.concatenate((memval,semval,motval,glaval))
        Xtrain_rest=np.reshape(Xtrain_rest,(-1,number))
        Xval_rest=np.reshape(Xval_rest,(-1,number))
        ytrain_task = np.ones(Xtrain_task.shape[0], dtype = int)
        ytrain_rest=np.zeros(Xtrain_rest.shape[0], dtype=int)
        yval_task = np.ones(Xval_task.shape[0], dtype = int)
        yval_rest=np.zeros(Xval_rest.shape[0], dtype=int)
        X_tr=np.concatenate((Xtrain_task, Xtrain_rest))
        y_tr = np.concatenate((ytrain_task,ytrain_rest))
        Xval=np.concatenate((Xval_task,Xval_rest))
        yval=np.concatenate((yval_task,yval_rest))
        scaler = preprocessing.StandardScaler().fit(X_tr)
        scaler.transform(X_tr)
        clf.fit(X_tr,y_tr)
        scaler.transform(Xval)
        CV_score=clf.score(Xval, yval)
        CVacc.append(CV_score)
        scaler.transform(Xtest)
        score=clf.score(Xtest, ytest)
        DSacc.append(score)
    same_sub=mean(CVacc)
    diff_sub=mean(DSacc)
    return same_sub, diff_sub

def runScript():
    DS_df=pd.DataFrame()
    ALL_df=pd.DataFrame()
    for i in netRoi:
        idx=netRoi[i]
        DS=DSmodel(idx)
        DS['feature']=i
        DS['Network']=idx
        DS_df=pd.concat([DS_df,DS])
        ALL=modelAll(idx)
        ALL['feature']=number
        ALL_df=pd.concat([ALL_df,ALL])
        print('Finished with '+i)
    DS_df.to_csv(outDir+'single_task/wholeNet_acc.csv', index=False)
    ALL_df.to_csv(outDir+'ALL_Binary/wholeNet_acc.csv', index=False)

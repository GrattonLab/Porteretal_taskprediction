#!/usr/bin/env python
# coding: utf-8

# In[ ]:
from sklearn import preprocessing
from sklearn.model_selection import LeaveOneOut
from sklearn.linear_model import RidgeClassifier
import numpy as np
import reshape
import pandas as pd
import itertools
import scipy.io
import random
import os
import sys
from statistics import mean
#import other python scripts for further anlaysis
# Initialization of directory information:
#thisDir = os.path.expanduser('~/Desktop/MSC_Alexis/analysis/')
thisDir = os.path.expanduser('~/Desktop/Porteretal_taskprediction/')
dataDir = thisDir + 'data/corrmats/'
outDir = thisDir + 'output/results/Ridge/'
# Subjects and tasks
taskList=['semantic','glass', 'motor','mem']
subList=['MSC01','MSC02','MSC03','MSC04','MSC05','MSC06','MSC07','MSC10']
subsComb=(list(itertools.permutations(subList, 2)))
def rdmNet_DS():
    DS_df=pd.DataFrame()
    featureSize=np.logspace(1, 4.7, num=39,dtype=int)
    for i in range(100):
        for number in featureSize:
            #generate a new index
            idx=np.random.randint(55278, size=(number))
            DS=DSmodel(idx)
            DS['feature']=number
            DS_df=pd.concat([DS_df,DS])
        print('Finished with '+str(i)+' in iteration 100')
    DS_df.to_csv(outDir+'single_task/rdmNet_acc.csv', index=False)



def rdmNet_classifyAll():
    featureSize=np.logspace(1, 4.7, num=39,dtype=int)
    ALL_df=pd.DataFrame()
    for number in featureSize:
        for i in range(100):
            #generate a new index
            idx=np.random.randint(55278, size=(number))
            ALL=modelAll(idx, number)
            ALL_df=pd.concat([ALL_df,ALL])
        print('Finished with '+str(i)+ ' log sample out of 39')
    ALL_df.to_csv(outDir+'ALL_Binary/rdmNet_acc.csv', index=False)

def AllSubFiles(test_sub,feat):
    """
    Return task and rest FC all subs
    Parameters
    -----------
    test_sub: Array of testing subs
    Returns
    ------------
    taskFC, restFC : Array of task and rest FC of all testing subs
    """
    a_memFC=reshape.randFeats(dataDir+'mem/'+test_sub[0]+'_parcel_corrmat.mat',feat)
    a_semFC=reshape.randFeats(dataDir+'semantic/'+test_sub[0]+'_parcel_corrmat.mat',feat)
    a_glassFC=reshape.randFeats(dataDir+'glass/'+test_sub[0]+'_parcel_corrmat.mat',feat)
    a_motFC=reshape.randFeats(dataDir+'motor/'+test_sub[0]+'_parcel_corrmat.mat',feat)
    a_restFC=reshape.randFeats(dataDir+'rest/corrmats_timesplit/fourths/'+test_sub[0]+'_parcel_corrmat.mat',feat)

    b_memFC=reshape.randFeats(dataDir+'mem/'+test_sub[1]+'_parcel_corrmat.mat',feat)
    b_semFC=reshape.randFeats(dataDir+'semantic/'+test_sub[1]+'_parcel_corrmat.mat',feat)
    b_glassFC=reshape.randFeats(dataDir+'glass/'+test_sub[1]+'_parcel_corrmat.mat',feat)
    b_motFC=reshape.randFeats(dataDir+'motor/'+test_sub[1]+'_parcel_corrmat.mat',feat)
    b_restFC=reshape.randFeats(dataDir+'rest/corrmats_timesplit/fourths/'+test_sub[1]+'_parcel_corrmat.mat',feat)

    c_memFC=reshape.randFeats(dataDir+'mem/'+test_sub[2]+'_parcel_corrmat.mat',feat)
    c_semFC=reshape.randFeats(dataDir+'semantic/'+test_sub[2]+'_parcel_corrmat.mat',feat)
    c_glassFC=reshape.randFeats(dataDir+'glass/'+test_sub[2]+'_parcel_corrmat.mat',feat)
    c_motFC=reshape.randFeats(dataDir+'motor/'+test_sub[2]+'_parcel_corrmat.mat',feat)
    c_restFC=reshape.randFeats(dataDir+'rest/corrmats_timesplit/fourths/'+test_sub[2]+'_parcel_corrmat.mat',feat)

    d_memFC=reshape.randFeats(dataDir+'mem/'+test_sub[3]+'_parcel_corrmat.mat',feat)
    d_semFC=reshape.randFeats(dataDir+'semantic/'+test_sub[3]+'_parcel_corrmat.mat',feat)
    d_glassFC=reshape.randFeats(dataDir+'glass/'+test_sub[3]+'_parcel_corrmat.mat',feat)
    d_motFC=reshape.randFeats(dataDir+'motor/'+test_sub[3]+'_parcel_corrmat.mat',feat)
    d_restFC=reshape.randFeats(dataDir+'rest/corrmats_timesplit/fourths/'+test_sub[3]+'_parcel_corrmat.mat',feat)

    e_memFC=reshape.randFeats(dataDir+'mem/'+test_sub[4]+'_parcel_corrmat.mat',feat)
    e_semFC=reshape.randFeats(dataDir+'semantic/'+test_sub[4]+'_parcel_corrmat.mat',feat)
    e_glassFC=reshape.randFeats(dataDir+'glass/'+test_sub[4]+'_parcel_corrmat.mat',feat)
    e_motFC=reshape.randFeats(dataDir+'motor/'+test_sub[4]+'_parcel_corrmat.mat',feat)
    e_restFC=reshape.randFeats(dataDir+'rest/corrmats_timesplit/fourths/'+test_sub[4]+'_parcel_corrmat.mat',feat)

    f_memFC=reshape.randFeats(dataDir+'mem/'+test_sub[5]+'_parcel_corrmat.mat',feat)
    f_semFC=reshape.randFeats(dataDir+'semantic/'+test_sub[5]+'_parcel_corrmat.mat',feat)
    f_glassFC=reshape.randFeats(dataDir+'glass/'+test_sub[5]+'_parcel_corrmat.mat',feat)
    f_motFC=reshape.randFeats(dataDir+'motor/'+test_sub[5]+'_parcel_corrmat.mat',feat)
    f_restFC=reshape.randFeats(dataDir+'rest/corrmats_timesplit/fourths/'+test_sub[5]+'_parcel_corrmat.mat',feat)

    g_memFC=reshape.randFeats(dataDir+'mem/'+test_sub[6]+'_parcel_corrmat.mat',feat)
    g_semFC=reshape.randFeats(dataDir+'semantic/'+test_sub[6]+'_parcel_corrmat.mat',feat)
    g_glassFC=reshape.randFeats(dataDir+'glass/'+test_sub[6]+'_parcel_corrmat.mat',feat)
    g_motFC=reshape.randFeats(dataDir+'motor/'+test_sub[6]+'_parcel_corrmat.mat',feat)
    g_restFC=reshape.randFeats(dataDir+'rest/corrmats_timesplit/fourths/'+test_sub[6]+'_parcel_corrmat.mat',feat)


    taskFC=np.concatenate((a_memFC,a_semFC,a_glassFC,a_motFC,b_memFC,b_semFC,b_glassFC,b_motFC,c_memFC,c_semFC,c_glassFC,c_motFC,d_memFC,d_semFC,d_glassFC,d_motFC,e_memFC,e_semFC,e_glassFC,e_motFC,f_memFC,f_semFC,f_glassFC,f_motFC,g_memFC,g_semFC,g_glassFC,g_motFC))
    restFC=np.concatenate((a_restFC,b_restFC,c_restFC,d_restFC,e_restFC,f_restFC,g_restFC))

    return taskFC, restFC



def modelAll(feat,number):
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
    #clf=RidgeClassifier()
    #train sub
    clf=RidgeClassifier(max_iter=10000)
    master_df=pd.DataFrame()
    data=np.array(['MSC01','MSC02','MSC03','MSC04','MSC05','MSC06','MSC07','MSC10'],dtype='<U61')
    loo = LeaveOneOut()
    for  test, train in loo.split(data): #train on one sub test on the rest
        tmp=pd.DataFrame()
        train_sub=data[train]
        test_sub=data[test]
    #train sub
        memFC=reshape.randFeats(dataDir+'mem/'+train_sub[0]+'_parcel_corrmat.mat',feat)
        semFC=reshape.randFeats(dataDir+'semantic/'+train_sub[0]+'_parcel_corrmat.mat',feat)
        glassFC=reshape.randFeats(dataDir+'glass/'+train_sub[0]+'_parcel_corrmat.mat',feat)
        motFC=reshape.randFeats(dataDir+'motor/'+train_sub[0]+'_parcel_corrmat.mat',feat)
        restFC=reshape.randFeats(dataDir+'rest/corrmats_timesplit/fourths/'+train_sub[0]+'_parcel_corrmat.mat',feat) #keep tasks seperated in order to collect the right amount of days
        nsize=restFC.shape[1]
        restFC=np.reshape(restFC,(10,4,nsize)) #reshape to gather correct days
        #test sub
        test_taskFC,test_restFC=AllSubFiles(test_sub,feat)
        diff_score, same_score=folds(train_sub, clf, memFC,semFC,glassFC,motFC, restFC, test_taskFC,test_restFC)
        tmp['train']=train_sub
        tmp['same_sub']=same_score
        tmp['diff_sub']=diff_score
        tmp['feat']=number
        master_df=pd.concat([master_df,tmp])
    return master_df

def folds(train_sub, clf, memFC,semFC,glassFC,motFC, restFC, test_taskFC,test_restFC):
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
    X_te=np.concatenate((test_taskFC, test_restFC))
    y_te=np.concatenate((testT, testR))
    CVacc=[]
    DSacc=[]
    if train_sub=='MSC03':
        split=np.empty((8,number))
    elif train_sub=='MSC06' or train_sub=='MSC07':
        split=np.empty((9,number))
    else:
        split=np.empty((10,number))
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
        X_val=np.concatenate((Xval_task, Xval_rest))
        y_tr = np.concatenate((ytrain_task,ytrain_rest))
        y_val=np.concatenate((yval_task, yval_rest))
        scaler = preprocessing.StandardScaler().fit(X_tr)
        scaler.transform(X_tr)
        clf.fit(X_tr,y_tr)
        scaler.transform(X_val)
        CV_score=clf.score(X_val, y_val)
        CVacc.append(CV_score)
        scaler.transform(X_te)
        score=clf.score(X_te,y_te)
        DSacc.append(score)
    diff_sub=mean(DSacc)
    same_sub=mean(CVacc)
    return diff_sub, same_sub



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

    clf=RidgeClassifier()
    DS=pd.DataFrame()
    data=np.array(['MSC01','MSC02','MSC03','MSC04','MSC05','MSC06','MSC07','MSC10'],dtype='<U61')
    loo = LeaveOneOut()
    for t in taskList:
        for  test, train in loo.split(data): #train on one sub test on the rest
            tmp=pd.DataFrame()
            train_sub=data[train]
            test_sub=data[test]
            taskFC=reshape.randFeats(dataDir+t+'/'+train_sub[0]+'_parcel_corrmat.mat',idx)
            restFC=reshape.randFeats(dataDir+'rest/'+train_sub[0]+'_parcel_corrmat.mat',idx) #keep tasks seperated in order to collect the right amount of days
            test_taskFC,test_restFC=AllSubFiles_DS(test_sub,t,idx)
            same_sub, diff_sub=single_task_folds(clf, taskFC,restFC, test_taskFC,test_restFC)
            tmp['train']=train_sub
            tmp['task']=t
            tmp['same_sub']=same_sub
            tmp['diff_sub']=diff_sub
            DS=pd.concat([DS,tmp])
    return DS

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
    a_taskFC=reshape.randFeats(dataDir+task+'/'+test_sub[0]+'_parcel_corrmat.mat',idx)
    a_restFC=reshape.randFeats(dataDir+'rest/'+test_sub[0]+'_parcel_corrmat.mat',idx)

    b_taskFC=reshape.randFeats(dataDir+task+'/'+test_sub[1]+'_parcel_corrmat.mat',idx)
    b_restFC=reshape.randFeats(dataDir+'rest/'+test_sub[1]+'_parcel_corrmat.mat',idx)

    c_taskFC=reshape.randFeats(dataDir+task+'/'+test_sub[2]+'_parcel_corrmat.mat',idx)
    c_restFC=reshape.randFeats(dataDir+'rest/'+test_sub[2]+'_parcel_corrmat.mat',idx)

    d_taskFC=reshape.randFeats(dataDir+task+'/'+test_sub[3]+'_parcel_corrmat.mat',idx)
    d_restFC=reshape.randFeats(dataDir+'rest/'+test_sub[3]+'_parcel_corrmat.mat',idx)

    e_taskFC=reshape.randFeats(dataDir+task+'/'+test_sub[4]+'_parcel_corrmat.mat',idx)
    e_restFC=reshape.randFeats(dataDir+'rest/'+test_sub[4]+'_parcel_corrmat.mat',idx)

    f_taskFC=reshape.randFeats(dataDir+task+'/'+test_sub[5]+'_parcel_corrmat.mat',idx)
    f_restFC=reshape.randFeats(dataDir+'rest/'+test_sub[5]+'_parcel_corrmat.mat',idx)

    g_taskFC=reshape.randFeats(dataDir+task+'/'+test_sub[6]+'_parcel_corrmat.mat',idx)
    g_restFC=reshape.randFeats(dataDir+'rest/'+test_sub[6]+'_parcel_corrmat.mat',idx)


    taskFC=np.concatenate((a_taskFC,b_taskFC,c_taskFC,d_taskFC,e_taskFC,f_taskFC,g_taskFC))
    restFC=np.concatenate((a_restFC,b_restFC,c_restFC,d_restFC,e_restFC,f_restFC,g_restFC))

    return taskFC, restFC


def single_task_folds(clf,taskFC, restFC, test_taskFC, test_restFC):
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

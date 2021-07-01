#!/usr/bin/env python
# coding: utf-8

# In[ ]:

from sklearn.model_selection import train_test_split
import matlab.engine
import numpy as np
from sklearn.model_selection import LeaveOneOut
from sklearn.linear_model import RidgeClassifier
import os
import sys
import pandas as pd
from sklearn.model_selection import cross_val_score
import itertools
import reshape
import warnings
import random
warnings.filterwarnings("ignore")
# Initialization of directory information:
thisDir = os.path.expanduser('~/Desktop/MSC_Alexis/analysis/')
dataDir=thisDir+'data/mvpa_data/'
# Subjects and tasks
framesList=[50,100,150,200,250,300]
subList=['MSC02','MSC05','MSC06','MSC07']

#all possible combinations of subs
subsComb=(list(itertools.permutations(subList, 2)))
#this function runs a matlab script that randomly selects frames in chunks and converts them into a numpy array
def getFrames(sub='MSC01', num=5, task='mem'):
    nrois=333
    eng=matlab.engine.start_matlab()
    parcel=eng.reframe(sub, num, task)
    fileFC=np.asarray(parcel)
    #Replace nans and infs with zero
    fileFC=np.nan_to_num(fileFC)
    if fileFC.ndim<3:
        return np.empty([])
    else:
        nsess=fileFC.shape[2]
    #Consistent parameters to use for editing datasets
    #Index upper triangle of matrix
        mask=np.triu_indices(nrois,1)
        ds=np.empty((nsess, int(nrois*(nrois-1)/2)))
        count=0
    #Loop through all 10 days to reshape correlations into linear form
        for sess in range(nsess):
            tmp=fileFC[:,:,sess]
            ds[count]=tmp[mask]
            count=count+1
    return ds


def storeResults():
    CV_permutations=pd.DataFrame()
    BS_permutations=pd.DataFrame()
    SS_permutations=pd.DataFrame()
    DS_permutations=pd.DataFrame()
    count=0
    featureSize=np.logspace(1.7,2.54, num=1000,dtype=int)
    featureSize=featureSize.tolist()
    for i in featureSize:
        CV=classifyCV(i)
        BS=classifyBS(i)
        SS=classifySS(i)
        DS=classifyDS(i)
        print(count)
        count=count+1
        CV_permutations=pd.concat([CV_permutations,CV])
        BS_permutations=pd.concat([BS_permutations,BS])
        SS_permutations=pd.concat([SS_permutations,SS])
        DS_permutations=pd.concat([DS_permutations,DS])
    CV_permutations.to_csv(thisDir+'output/results/permutation/CV/frames.csv',index=False)
    BS_permutations.to_csv(thisDir+'output/results/permutation/BS/frames.csv',index=False)
    SS_permutations.to_csv(thisDir+'output/results/permutation/SS/frames.csv',index=False)
    DS_permutations.to_csv(thisDir+'output/results/permutation/DS/frames.csv',index=False)


def classifyDS(num):
    """
    Classifying different subjects (DS) along the same task

    Parameters
    -------------

    Returns
    -------------
    dfDS : DataFrame
        Dataframe consisting of average accuracy across all subjects

    """
    df=pd.DataFrame()
    dfDS=pd.DataFrame(subsComb,columns=['train_sub','test_sub'])
    dfDS['train_task']='mem'
    dfDS['test_task']='mem'
    acc_scores_per_task=[]
    for index, row in dfDS.iterrows():
        score=model('DS',num,train_sub=row['train_sub'], test_sub=row['test_sub'], train_task='mem', test_task='mem')
        acc_scores_per_task.append(score)
    dfDS['acc']=acc_scores_per_task
    dfDS['frames']=num
    df=pd.concat([df,dfDS])
    return df
def classifySS(num):
    """
    Classifying the same subject (SS) along a different task

    Parameters
    -------------


    Returns
    -------------
    dfSS : DataFrame
        Dataframe consisting of average accuracy across all subjects

    """
    df=pd.DataFrame()
    dfSS=pd.DataFrame(subList, columns=['sub'])
    dfSS['train_task']='mem'
    dfSS['test_task']='mixed'
    acc_scores_per_task=[]
    for sub in subList:
        score=model('SS', num, train_sub=sub, test_sub=sub, train_task='mem', test_task='mixed')
        acc_scores_per_task.append(score)
    dfSS['acc']=acc_scores_per_task
    dfSS['frames']=num
    df=pd.concat([df,dfSS])
    #save accuracy
    return df
def classifyBS(num):
    """
    Classifying different subjects (BS) along different tasks

    Parameters
    -------------


    Returns
    -------------
    dfBS : DataFrame
        Dataframe consisting of average accuracy across all subjects

    """
    df=pd.DataFrame()
    acc_scores_per_task=[]
    dfBS=pd.DataFrame(subsComb,columns=['train_sub','test_sub'])
    dfBS['train_task']='mem'
    dfBS['test_task']='mixed'
    for index, row in dfBS.iterrows():
        score=model('BS', num, train_sub=row['train_sub'], test_sub=row['test_sub'], train_task='mem', test_task='mixed')
        acc_scores_per_task.append(score)
    dfBS['acc']=acc_scores_per_task
    dfBS['frames']=num
    df=pd.concat([df,dfBS])
    #save accuracy
    return df

def classifyCV(num):
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
    acc_scores_per_task=[]
    subs_per_task=[]
    task_per_task=[]
    frames_per_task=[]
    for sub in subList:
        taskFC=getFrames(sub,num,'mem')
        restFC=getFrames(sub,num, 'rest')
        if taskFC.size==1 or restFC.size==1:
            mu='9999'
            print('subject '+sub+ ' has no data for frame '+str(num))
        else:
            folds=taskFC.shape[0]
            x_train, y_train=reshape.concateFC(taskFC, restFC)
            CVscores=cross_val_score(clf, x_train, y_train, cv=folds)
            mu=CVscores.mean()
            acc_scores_per_task.append(mu)
            subs_per_task.append(sub)
            frames_per_task.append(num)
    dfCV=pd.DataFrame({'train_sub':subs_per_task,'train_task':'mem', 'acc':acc_scores_per_task, 'frames':frames_per_task})
    return dfCV
def model(analysis, num,train_sub, test_sub, train_task, test_task):
    """
    Preparing machine learning model with appropriate data

    Parameters
    -------------
    classifier : str
            The statistical method used for classification
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
    df=pd.DataFrame()
    taskFC=getFrames(train_sub,num,train_task)
    restFC=getFrames(train_sub,num, 'rest')
    #if the file returns a 1 dimensional empty array there weren't enough frames for sub
    if taskFC.size==1 or restFC.size==1:
        total_score='9999'
        print('subject '+train_sub+ ' does not have data for frame '+str(num))
    #if either rest or task FC has less than 5 samples don't use
    elif taskFC.shape[0]<5 or restFC.shape[0]<5:
        total_score='9999'
        print('subject '+train_sub+ ' does not have days for frame '+str(num))
    else:
    #if your subs are the same
        if train_sub==test_sub:
            test_taskFC=getFrames(test_sub, num, test_task)
            total_score=CV_folds(clf, analysis, taskFC, restFC, test_taskFC, restFC)
        else:
            test_taskFC=getFrames(test_sub, num, test_task)
            test_restFC=getFrames(test_sub, num, 'rest')
            total_score=CV_folds(clf, analysis, taskFC, restFC, test_taskFC, test_restFC)
    return total_score
#Calculate acc of cross validation within sub within task
def CV_folds(clf, analysis, taskFC, restFC, test_taskFC, test_restFC):
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
    if analysis=='SS':
        df=pd.DataFrame()
        acc_score=[]
        for train_index, test_index in loo.split(taskFC):
            Xtrain_rest, Xtest_rest=restFC[train_index], restFC[test_index]
            Xtrain_task=taskFC[train_index]
            ytrain_rest=r[train_index]
            ytrain_task=t[train_index]
            X_tr=np.concatenate((Xtrain_task, Xtrain_rest))
            y_tr = np.concatenate((ytrain_task,ytrain_rest))
            clf.fit(X_tr,y_tr)
            tmpdf=pd.DataFrame()
            acc_scores_per_fold=[]
            for t_index, te_index in loo.split(test_taskFC):
                Xtest_task=test_taskFC[te_index]
                X_Test = np.concatenate((Xtest_task, Xtest_rest))
                y_Test = np.array([1, 0])
                #test set
                clf.predict(X_Test)
                #Get accuracy of model
                ACCscores=clf.score(X_Test,y_Test)
                acc_scores_per_fold.append(ACCscores)
            tmpdf['inner_fold']=acc_scores_per_fold
            score=tmpdf['inner_fold'].mean()
            acc_score.append(score)
        df['outer_fold']=acc_score
        total_score=df['outer_fold'].mean()
    else:
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

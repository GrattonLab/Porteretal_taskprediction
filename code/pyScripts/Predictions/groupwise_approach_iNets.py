#!/usr/bin/env python
# coding: utf-8

# In[ ]:
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import LeaveOneOut
from sklearn.linear_model import RidgeClassifier
import numpy as np
import pandas as pd
import glob
from sklearn.metrics import confusion_matrix
from statistics import mean
import scipy.io
import os
import sys
# Initialization of directory information:
thisDir = '/projects/b1081/member_directories/aporter/TaskRegScripts/SurfaceResiduals/'
dataDir = thisDir
outDir = thisDir + 'output/'
# Subjects and tasks
#thisDir = os.path.expanduser('~/Desktop/Porteretal_taskprediction/')
#dataDir = thisDir + 'data/corrmats/iNetworks/'
#outDir = thisDir + 'output/results/Ridge/'
taskList=['slowreveal','mixed']
sesList = ['ses-1','ses-2','ses-3','ses-4']



def model(train_task):
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
            Average accuracy of all folds leave one sub out of a given task
    within_sens : float
            Sensitivity score for model within task
    within spec : float
            Specificity score for model within task
    btn_score : float
            Average accuracy of all folds leave one sub out of a given task
    btn_sens : float
            Sensitivity score for model within task
    btn spec : float
            Specificity score for model within task

    """

    clf=RidgeClassifier()
    loo = LeaveOneOut()
    SSacc_folds=[]
    dfDS=pd.DataFrame()
    #get all subs for a given task
    #make list based on available subjects in task folder
    subList=[]
    pattern = dataDir+train_task+"/*"
    files = [os.path.basename(x) for x in glob.glob(pattern)]
    for i in files:
        subList.append(i.split('_', 1)[0])
    #Remove duplicates
    subList = list(set(subList))
    data=np.array(subList,dtype='<U61')
    sessions=np.array(sesList,dtype='<U61')
    for  train, test in loo.split(data): #test on all other subj
        train_sub = data[test] #testing subj
        test_subs = data[train] #training subjs
        #randomly collects one run one session per subj
        testing_set_task = iNets_OS_onerun(train_task, test_subs)
        testing_set_rest = iNets_OS_onerun('rest', test_subs)
        #make sure rest and task sets are the same amount
        y_test_task=np.ones(testing_set_task.shape[0])
        y_test_rest=np.zeros(testing_set_rest.shape[0])
        ytest=np.concatenate((y_test_task,y_test_rest))
        Xtest=np.concatenate((testing_set_task,testing_set_rest))
        #left out sub
        LOS_task=iNetOpenRun(train_task, train_sub[0])
        LOS_rest=iNetOpenRun('rest', train_sub[0])
        task=np.ones(LOS_task.shape[0])
        rest=np.zeros(LOS_rest.shape[0])
        y=np.concatenate((task,rest))
        X=np.concatenate((LOS_task,LOS_rest))
        clf.fit(Xtest,ytest)
        #Same subject
        SSscores=clf.score(X,y)
        SSacc_folds.append(SSscores)

    #SStotal_acc=(mean(SSacc_folds)
    dfDS['acc'] = SSacc_folds
    dfDS.to_csv(outDir+train_task+'_iNets_acc_groupwise.csv',index=False)


def iNets_OS_onerun(train_task, test_subs):
    """
    Calculate FC matrices for all subs in a given task keeping one run randomly
    Parameters
    -----------
    df : str
        Path to file
    Returns
    -----------
    ds : 2D upper triangle FC measures in (roi, days) format

    """
    result_arr = []
    #loop through and append all test sets do each task/rest separate
    for sub in test_subs:
        all_sub_mats = iNetOpenRun(train_task, sub)
        result_arr.append(all_sub_mats)
    result_arr = np.concatenate(result_arr)
    return result_arr

def iNetOpenRun(train_task, sub):
    """
    Convert matlab files into upper triangle np.arrays for a given sub (one run)
    Parameters
    -----------
    df : str
        Path to file
    Returns
    -----------
    ds : 2D upper triangle FC measures in (roi, days) format

    """
    #find all files
    files = glob.glob(dataDir+train_task+'/'+sub+'_ses-*')
    nsess = len(files) #FC matrices
    nrois=333
    ds=np.empty((nsess, int(nrois*(nrois-1)/2)))
    count=0
    for f in files:
    #Consistent parameters to use for editing datasets

        #Load FC file
        fileFC=scipy.io.loadmat(f)

        #Convert to numpy array
        fileFC=np.array(fileFC['parcel_corrmat'])
        #Replace nans and infs with zero
        fileFC=np.nan_to_num(fileFC)
        #Index upper triangle of matrix
        mask=np.triu_indices(nrois,1)

    #Loop through all 10 days to reshape correlations into linear form

        tmp=fileFC[:,:]
        ds[count]=tmp[mask]
        count=count+1
    mask = (ds == 0).all(1)
    column_indices = np.where(mask)[0]
    df = ds[~mask,:]
    #randomly choose 1 run from 1 session
    df=df[np.random.choice(df.shape[0], 1, replace=False), :]
    return df

#for task in taskList:
#    model(task)

def MCmodel():
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
            Average accuracy of all folds leave one sub out of a given task
    within_sens : float
            Sensitivity score for model within task
    within spec : float
            Specificity score for model within task
    btn_score : float
            Average accuracy of all folds leave one sub out of a given task
    btn_sens : float
            Sensitivity score for model within task
    btn spec : float
            Specificity score for model within task

    """

    clf=RidgeClassifier()
    loo = LeaveOneOut()
    SSacc_folds=[]
    dfDS=pd.DataFrame()
    #get all subs for a given task
    #make list based on available subjects in task folder
    subList=[]
    pattern = dataDir+"mixed/*"
    files = [os.path.basename(x) for x in glob.glob(pattern)]
    for i in files:
        subList.append(i.split('_', 1)[0])
    #Remove duplicates
    subList = list(set(subList))
    data=np.array(subList,dtype='<U61')
    sessions=np.array(sesList,dtype='<U61')
    for  train, test in loo.split(data): #test on all other subj
        train_sub = data[test] #testing subj
        test_subs = data[train] #training subjs
        #randomly collects one run one session per subj
        testing_set_mix = iNets_OS_onerun('mixed', test_subs)
        testing_set_slow = iNets_OS_onerun('slowreveal', test_subs)
        testing_set_rest = iNets_OS_onerun('rest', test_subs)
        #make sure rest and task sets are the same amount
        y_test_slow=np.full(testing_set_slow.shape[0],2)
        y_test_mix=np.ones(testing_set_mix.shape[0])
        y_test_rest=np.zeros(testing_set_rest.shape[0])
        ytest=np.concatenate((y_test_slow,y_test_mix,y_test_rest))
        Xtest=np.concatenate((testing_set_slow,testing_set_mix,testing_set_rest))
        #left out sub
        LOS_slow=iNetOpenRun('slowreveal', train_sub[0])
        LOS_mix=iNetOpenRun('mixed', train_sub[0])
        LOS_rest=iNetOpenRun('rest', train_sub[0])
        slow=np.full(LOS_slow.shape[0],2)
        mix=np.ones(LOS_mix.shape[0])
        rest=np.zeros(LOS_rest.shape[0])
        y=np.concatenate((slow, mix,rest))
        X=np.concatenate((LOS_slow,LOS_mix,LOS_rest))
        clf.fit(Xtest,ytest)
        #Same subject
        SSscores=clf.score(X,y)
        SSacc_folds.append(SSscores)

    #SStotal_acc=(mean(SSacc_folds)
    dfDS['acc'] = SSacc_folds
    dfDS.to_csv(outDir+'MC_iNets_acc_groupwise.csv',index=False)


def Binarymodel():
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
            Average accuracy of all folds leave one sub out of a given task
    within_sens : float
            Sensitivity score for model within task
    within spec : float
            Specificity score for model within task
    btn_score : float
            Average accuracy of all folds leave one sub out of a given task
    btn_sens : float
            Sensitivity score for model within task
    btn spec : float
            Specificity score for model within task

    """

    clf=RidgeClassifier()
    loo = LeaveOneOut()
    SSacc_folds=[]
    dfDS=pd.DataFrame()
    #get all subs for a given task
    #make list based on available subjects in task folder
    subList=[]
    pattern = dataDir+"mixed/*"
    files = [os.path.basename(x) for x in glob.glob(pattern)]
    for i in files:
        subList.append(i.split('_', 1)[0])
    #Remove duplicates
    subList = list(set(subList))
    data=np.array(subList,dtype='<U61')
    sessions=np.array(sesList,dtype='<U61')
    for  train, test in loo.split(data): #test on all other subj
        train_sub = data[test] #testing subj
        test_subs = data[train] #training subjs
        #randomly collects one run one session per subj
        testing_set_mix = iNets_OS_onerun('mixed', test_subs)
        testing_set_slow = iNets_OS_onerun('slowreveal', test_subs)
        testing_set_rest = iNets_OS_onerun('rest', test_subs)
        #make sure rest and task sets are the same amount
        y_test_slow=np.ones(testing_set_slow.shape[0])
        y_test_mix=np.ones(testing_set_mix.shape[0])
        y_test_rest=np.zeros(testing_set_rest.shape[0])
        ytest=np.concatenate((y_test_slow,y_test_mix,y_test_rest))
        Xtest=np.concatenate((testing_set_slow,testing_set_mix,testing_set_rest))
        #left out sub
        LOS_slow=iNetOpenRun('slowreveal', train_sub[0])
        LOS_mix=iNetOpenRun('mixed', train_sub[0])
        LOS_rest=iNetOpenRun('rest', train_sub[0])
        slow=np.ones(LOS_slow.shape[0])
        mix=np.ones(LOS_mix.shape[0])
        rest=np.zeros(LOS_rest.shape[0])
        y=np.concatenate((slow, mix,rest))
        X=np.concatenate((LOS_slow,LOS_mix,LOS_rest))
        clf.fit(Xtest,ytest)
        #Same subject
        SSscores=clf.score(X,y)
        SSacc_folds.append(SSscores)

    #SStotal_acc=(mean(SSacc_folds)
    dfDS['acc'] = SSacc_folds
    dfDS.to_csv(outDir+'Binary_iNets_acc_groupwise.csv',index=False)

for task in taskList:
    model(task)
Binarymodel()
MCmodel()

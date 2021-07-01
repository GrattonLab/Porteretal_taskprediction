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
from sklearn.model_selection import KFold
import os
import sys
#import other python scripts for further anlaysis
# Initialization of directory information:
thisDir = os.path.expanduser('~/Desktop/MSC_Alexis/analysis/')
dataDir = thisDir + 'data/mvpa_data/'
outDir = thisDir + 'output/results/rdmNetwork/'
# Subjects and tasks
taskList=['semantic','glass', 'motor','mem']
subList=['MSC01','MSC02','MSC03','MSC04','MSC05','MSC06','MSC07','MSC10']
subsComb=(list(itertools.permutations(subList, 2)))


netRoi=dict([('unassign',14808),('default', 10824),('visual',8736),('fp', 4620),('dan',5264),('van',3151),('salience', 484),('co', 4060),('sm', 2375),('sm-lat', 316),('auditory', 564),('pmn',45),('pon',21),('co_co',780),('fp_co',960),('default_co',1640)])

#generate log sample
#1000 points for log selection
#loop through 125 times to generate 8*125=1000 samples per log point
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
    a_memFC=randFeats(dataDir+'mem/'+test_sub[0]+'_parcel_corrmat.mat',feat)
    a_semFC=randFeats(dataDir+'semantic/'+test_sub[0]+'_parcel_corrmat.mat',feat)
    a_glassFC=randFeats(dataDir+'glass/'+test_sub[0]+'_parcel_corrmat.mat',feat)
    a_motFC=randFeats(dataDir+'motor/'+test_sub[0]+'_parcel_corrmat.mat',feat)
    a_restFC=randFeats(dataDir+'rest/corrmats_timesplit/fourths/'+test_sub[0]+'_parcel_corrmat.mat',feat)

    b_memFC=randFeats(dataDir+'mem/'+test_sub[1]+'_parcel_corrmat.mat',feat)
    b_semFC=randFeats(dataDir+'semantic/'+test_sub[1]+'_parcel_corrmat.mat',feat)
    b_glassFC=randFeats(dataDir+'glass/'+test_sub[1]+'_parcel_corrmat.mat',feat)
    b_motFC=randFeats(dataDir+'motor/'+test_sub[1]+'_parcel_corrmat.mat',feat)
    b_restFC=randFeats(dataDir+'rest/corrmats_timesplit/fourths/'+test_sub[1]+'_parcel_corrmat.mat',feat)

    c_memFC=randFeats(dataDir+'mem/'+test_sub[2]+'_parcel_corrmat.mat',feat)
    c_semFC=randFeats(dataDir+'semantic/'+test_sub[2]+'_parcel_corrmat.mat',feat)
    c_glassFC=randFeats(dataDir+'glass/'+test_sub[2]+'_parcel_corrmat.mat',feat)
    c_motFC=randFeats(dataDir+'motor/'+test_sub[2]+'_parcel_corrmat.mat',feat)
    c_restFC=randFeats(dataDir+'rest/corrmats_timesplit/fourths/'+test_sub[2]+'_parcel_corrmat.mat',feat)

    d_memFC=randFeats(dataDir+'mem/'+test_sub[3]+'_parcel_corrmat.mat',feat)
    d_semFC=randFeats(dataDir+'semantic/'+test_sub[3]+'_parcel_corrmat.mat',feat)
    d_glassFC=randFeats(dataDir+'glass/'+test_sub[3]+'_parcel_corrmat.mat',feat)
    d_motFC=randFeats(dataDir+'motor/'+test_sub[3]+'_parcel_corrmat.mat',feat)
    d_restFC=randFeats(dataDir+'rest/corrmats_timesplit/fourths/'+test_sub[3]+'_parcel_corrmat.mat',feat)

    e_memFC=randFeats(dataDir+'mem/'+test_sub[4]+'_parcel_corrmat.mat',feat)
    e_semFC=randFeats(dataDir+'semantic/'+test_sub[4]+'_parcel_corrmat.mat',feat)
    e_glassFC=randFeats(dataDir+'glass/'+test_sub[4]+'_parcel_corrmat.mat',feat)
    e_motFC=randFeats(dataDir+'motor/'+test_sub[4]+'_parcel_corrmat.mat',feat)
    e_restFC=randFeats(dataDir+'rest/corrmats_timesplit/fourths/'+test_sub[4]+'_parcel_corrmat.mat',feat)

    f_memFC=randFeats(dataDir+'mem/'+test_sub[5]+'_parcel_corrmat.mat',feat)
    f_semFC=randFeats(dataDir+'semantic/'+test_sub[5]+'_parcel_corrmat.mat',feat)
    f_glassFC=randFeats(dataDir+'glass/'+test_sub[5]+'_parcel_corrmat.mat',feat)
    f_motFC=randFeats(dataDir+'motor/'+test_sub[5]+'_parcel_corrmat.mat',feat)
    f_restFC=randFeats(dataDir+'rest/corrmats_timesplit/fourths/'+test_sub[5]+'_parcel_corrmat.mat',feat)

    g_memFC=randFeats(dataDir+'mem/'+test_sub[6]+'_parcel_corrmat.mat',feat)
    g_semFC=randFeats(dataDir+'semantic/'+test_sub[6]+'_parcel_corrmat.mat',feat)
    g_glassFC=randFeats(dataDir+'glass/'+test_sub[6]+'_parcel_corrmat.mat',feat)
    g_motFC=randFeats(dataDir+'motor/'+test_sub[6]+'_parcel_corrmat.mat',feat)
    g_restFC=randFeats(dataDir+'rest/corrmats_timesplit/fourths/'+test_sub[6]+'_parcel_corrmat.mat',feat)


    taskFC=np.concatenate((a_memFC,a_semFC,a_glassFC,a_motFC,b_memFC,b_semFC,b_glassFC,b_motFC,c_memFC,c_semFC,c_glassFC,c_motFC,d_memFC,d_semFC,d_glassFC,d_motFC,e_memFC,e_semFC,e_glassFC,e_motFC,f_memFC,f_semFC,f_glassFC,f_motFC,g_memFC,g_semFC,g_glassFC,g_motFC))
    restFC=np.concatenate((a_restFC,b_restFC,c_restFC,d_restFC,e_restFC,f_restFC,g_restFC))

    return taskFC, restFC


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
    clf=RidgeClassifier()
    master_df=pd.DataFrame()
    data=np.array(['MSC01','MSC02','MSC03','MSC04','MSC05','MSC06','MSC07','MSC10'],dtype='<U61')
    loo = LeaveOneOut()
    for  test, train in loo.split(data): #train on one sub test on the rest
        tmp=pd.DataFrame()
        train_sub=data[train]
        test_sub=data[test]
    #train sub
        memFC=randFeats(dataDir+'mem/'+train_sub[0]+'_parcel_corrmat.mat',feat)
        semFC=randFeats(dataDir+'semantic/'+train_sub[0]+'_parcel_corrmat.mat',feat)
        glassFC=randFeats(dataDir+'glass/'+train_sub[0]+'_parcel_corrmat.mat',feat)
        motFC=randFeats(dataDir+'motor/'+train_sub[0]+'_parcel_corrmat.mat',feat)
        restFC=randFeats(dataDir+'rest/corrmats_timesplit/fourths/'+train_sub[0]+'_parcel_corrmat.mat',feat) #keep tasks seperated in order to collect the right amount of days
        #taskFC=np.concatenate((memFC,semFC,glassFC,motFC))
        #to have more control over sessions
        #taskFC=np.dstack((memFC,semFC,glassFC,motFC))#10x55278x4
        restFC=np.reshape(restFC,(10,4,number)) #reshape to gather correct days

        test_taskFC,test_restFC=AllSubFiles(test_sub,feat)

        #return taskFC,restFC, test_taskFC,test_restFC
        diff_sub_score, same_sub_score=K_folds(train_sub, number, clf, memFC,semFC,glassFC,motFC, restFC, test_taskFC, test_restFC)
        tmp['train']=train_sub
        tmp['same_sub']=same_sub_score
        tmp['diff_sub']=diff_sub_score
        tmp['features']=number
        master_df=pd.concat([master_df,tmp])
    return master_df
def K_folds(train_sub, number,clf, memFC,semFC,glassFC,motFC,restFC, test_taskFC, test_restFC):
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

    kf = LeaveOneOut()
    """
    taskSize=taskFC.shape[0]
    restSize=restFC.shape[0]
    t = np.ones(taskSize, dtype = int)
    r=np.zeros(restSize, dtype=int)
    """
    test_taskSize=test_taskFC.shape[0]
    test_restSize=test_restFC.shape[0]
    test_t = np.ones(test_taskSize, dtype = int)
    test_r=np.zeros(test_restSize, dtype=int)
    Xtest=np.concatenate((test_taskFC,test_restFC))
    ytest=np.concatenate((test_t,test_r))
    CVacc=[]
    df=pd.DataFrame()
    DSacc=[]
    #fold each training set
    if train_sub=='MSC03':
        split=np.empty((8,number))
        #xtrainSize=24
        #xtestSize=4
    elif train_sub=='MSC06' or train_sub=='MSC07':
        split=np.empty((9,number))
    else:
        split=np.empty((10,number))
    for train_index, test_index in kf.split(split):
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
        clf.fit(X_tr,y_tr)
        CV_score=clf.score(X_val, y_val)
        CVacc.append(CV_score)
        score=clf.score(Xtest,ytest)
        DSacc.append(score)
    df['cv']=CVacc
    #Different sub outer acc
    df['ds']=DSacc
    same_sub_score=df['cv'].mean()
    diff_sub_score=df['ds'].mean()
    return diff_sub_score, same_sub_score
def runAll():
    ALL_df=pd.DataFrame()
    for nullNet in netRoi:
        number=netRoi[nullNet]
        for n in range(10):
            #generate a new index
            idx=np.random.randint(55278, size=(number))
            ALL=modelAll(idx, number)
            ALL['Null_Network']=nullNet
            ALL_df=pd.concat([ALL_df,ALL])
    ALL_df.to_csv(outDir+'ALL/nullNet_acc.csv', index=False)
